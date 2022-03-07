#!/usr/bin/env python
"""
Main BLENDS augmentation class.

Copyright (c) 2022 The University of Texas Southwestern Medical Center.
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

import os
import glob
import numpy as np
import nibabel as nib
from sklearn import datasets
from nipype.pipeline import Node, Workflow
from nipype.interfaces import fsl, image, ants
from . import transforms, skullstrip
import timeit
import tempfile

strT1TemplatePath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                               'data/mni_icbm152_t1_tal_nlin_sym_09a_brain.nii.gz'))
strEPITemplatePath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                               'data/tpl-MNI152NLin2009cAsym_res-02_desc-fMRIPrep_boldref_brain.nii.gz'))

def _distance_transform(tupShape, arrPoints):
    _, arrY, arrZ = np.mgrid[:tupShape[0], :tupShape[1], :tupShape[2]]
    # do not use x direction in order to enforce bilateral symmetry
    def _distance(arrGrid, arrLoc):
        nPoints = arrLoc.shape[0]
        arrGrid = np.tile(arrGrid[:, :, :, np.newaxis], [1, 1, 1, nPoints])
        return arrGrid - arrLoc[np.newaxis, np.newaxis, np.newaxis, :]

    arrDist = np.sqrt(_distance(arrY, arrPoints[:, 1]) ** 2
                      + _distance(arrZ, arrPoints[:, 2]) ** 2)
    arrDist = arrDist.min(axis=-1)
    return arrDist

class WarpBlender:
    def __init__(self, warp_source_dir, n_ants_jobs=8):
        """
        BLENDS augmentation class. Performs augmentation of brain MRI by
        applying a random spatially blended combination of precomputed nonlinear
        warps.

        Steps:
        1.  Use prepare_source() to skullstrip the input image and register it to the MNI template.
        2.  Use generate_warp() to compute the new spatially blended warp.
        3.  Call augment(). 

        If augmenting a pair of fMRI/sMRI from the same subject:
        1.  Create 2 WarpBlender objects and call prepare_source() on each image.
        2.  Call generate_warp() on one of them, then assign the same generated warp to the other
            WarpBlender using the arrWarpBlended attribute. E.g.::

                augmenterAnat = blending.WarpBlender(...)
                augmenterFunc = blending.WarpBlender(...)
                augmenterAnat.prepare_source(anat_path, registration='anat')
                augmenterFunc.prepare_source(func_path, registration='func')
                augmenter.Func.generate_warp(...)
                augmenterAnat.arrWarpBlended = augmenterFunc.arrWarpBlended

        Dependencies required on system PATH:
        * ANTs
        * ROBEX
        * FSL

        Args:
            warp_source_dir (str): path to a directory containing precomputed MNI->subject space warps in ANTs .nii or .nii.gz format.
            n_ants_jobs (int, optional): number of parallel ANTs threads. Defaults to 8.

        Raises:
            NotADirectoryError: warp_source_dir is not a valid directory
        """        
        if isinstance(warp_source_dir, list):
            self.warp_sources = warp_source_dir
        else:
            if os.path.isdir(warp_source_dir):
                self.warp_sources = sorted(glob.glob(os.path.join(warp_source_dir, '*.nii*')))
            else:
                raise NotADirectoryError(warp_source_dir + ' is not a valid directory')

        # Find warp dimensions
        img = nib.load(self.warp_sources[0])
        self.tupShape = img.shape

        self.n_ants_jobs = n_ants_jobs

        self.n_sources = None
        self.arrWarpBlended = None
        self.strTemplate = None
        self.strRegisteredPath = None
        self.bFunc = False
        # Store intermediate files in temporary directory. This is removed once this object is deleted.
        self.tempdir = tempfile.TemporaryDirectory()

    def _combine_warps(self, source_warps, blending_mats):
        lsWarps = [nib.load(warp).get_data() for warp in source_warps]
        arrWarps = np.stack(lsWarps, axis=0)
        arrWarpNew = arrWarps.squeeze() * blending_mats
        return arrWarpNew.sum(axis=0)

    def _skullstrip_register_func(self, input_img, output_path):
        output_path = os.path.abspath(output_path)
        workflow = Workflow('register_func', base_dir=output_path)

        # Compute mean image
        meanimage = Node(fsl.MeanImage(dimension='T', output_type='NIFTI_GZ'), name='mean')
        meanimage.inputs.in_file = input_img

        # Skull strip
        funcstrip = skullstrip.make_func_mask_workflow(base_dir=output_path)
        workflow.connect(meanimage, 'out_file', funcstrip, 'inputnode.mean_file')

        # Register with EPI template.
        register = Node(ants.Registration(fixed_image=strEPITemplatePath,
                            transforms=['Translation', 'Rigid', 'Affine', 'SyN'],
                            transform_parameters=[(0.1,), (0.1,), (0.1,), (0.2, 3.0, 0.0)],
                            number_of_iterations=([[1000, 1000, 1000]] * 3 + [[100, 50, 30]]),
                            dimension=3,
                            write_composite_transform=True,
                            collapse_output_transforms=False,
                            metric=['Mattes'] * 3 + [['Mattes', 'CC']],
                            metric_weight=[1, 1, 1, [0.5, 0.5]],
                            radius_or_number_of_bins=[32] * 3 + [[32, 4]],
                            sampling_strategy=['Regular'] * 3 + [[None, None]],
                            sampling_percentage=[0.3] * 3 + [[None, None]],
                            convergence_threshold=[1e-8] * 3 + [-0.01],
                            convergence_window_size=[20, 20, 20, 5],
                            smoothing_sigmas=[[4, 2, 1]] * 3 + [[1, 0.5, 0]],
                            sigma_units=['vox'] * 4,
                            shrink_factors=[[6, 4, 2], [3, 2, 1], [3, 2, 1], [4, 2, 1]],
                            use_estimate_learning_rate_once=[True] * 4,
                            use_histogram_matching=[False] * 3 + [True],
                            initial_moving_transform_com=1,
                            terminal_output='file',
                            num_threads=self.n_ants_jobs),
                        name='register',
                        mem_gb=16, n_procs=self.n_ants_jobs
                        )
        workflow.connect(funcstrip, 'outputnode.masked_file', register, 'moving_image')

        # Apply transformation to the entire image timeseries
        transform = Node(ants.ApplyTransforms(reference_image=strEPITemplatePath, input_image=input_img, float=True,
                                               output_image=os.path.join(output_path, 'registered_func_Warped.nii.gz'),
                                                interpolation='BSpline', interpolation_parameters=(5,),
                                                input_image_type=3, num_threads=self.n_ants_jobs),
                           name='apply_transforms')
        transform.interface.num_threads = self.n_ants_jobs
        workflow.connect(register, 'composite_transform', transform, 'transforms')

        workflow.run()
        return os.path.join(output_path, 'registered_func_Warped.nii.gz')


    def _skullstrip_register_anat(self, input_img, output_path):
        output_path = os.path.abspath(output_path)
        workflow = Workflow('register_anat', base_dir=output_path)

        # Skull strip
        anatstrip = Node(skullstrip.Robex(in_file=input_img), name='skullstrip')

        # Register to MNI template
        register = Node(ants.RegistrationSynQuick(fixed_image=strT1TemplatePath, num_threads=self.n_ants_jobs,
                                                  output_prefix=os.path.join(output_path, 'registered_anat_')),
                        name='register',
                        mem_gb=16, n_procs=self.n_ants_jobs
                        )
        workflow.connect(anatstrip, 'out_file', register, 'moving_image')
        workflow.run()
        return os.path.join(output_path, 'registered_anat_Warped.nii.gz')


    def prepare_source(self, input_img, registration='func'):
        """Skullstrip and register input image to MNI space in preparation for augmentation.

        Args:
            input_img (str): path to input image (.nii*)
            registration (str, optional): type of input image, 'anat' or 'func'. Defaults to 'func'.

        Raises:
            ValueError: registration is an invalid value
        """        
        # Affine register input image to MNI space
        if registration == 'func':
            self.strRegisteredPath = self._skullstrip_register_func(input_img, self.tempdir.name)
            self.strTemplate = strEPITemplatePath
            self.bFunc = True
        elif registration == 'anat':
            self.strRegisteredPath = self._skullstrip_register_anat(input_img, self.tempdir.name)
            self.strTemplate = strT1TemplatePath
            self.bFunc = False
        else:
            raise ValueError(registration + ' is not a valid registration mode')


    def generate_warp(self, n_sources=4, seed=None, verbose=False):
        """Generate a blended warp.

        Args:
            n_sources (int, optional): number of existing warps to combine. Defaults to 4.
            seed (int, optional): random seed. Defaults to None.
            verbose (bool, optional): print execution time. Defaults to False.
        """        
        self.n_sources = n_sources
        t0 = timeit.default_timer()
        # use make_blobs to generate isotropic clusters of points in space.
        arrPointsOrig, arrLabels = datasets.make_blobs(n_samples=100, n_features=3, centers=n_sources,
                                                       cluster_std=1,
                                                       random_state=seed)
        arrPoints = arrPointsOrig - arrPointsOrig.min(axis=0)
        arrPoints *= np.array(self.tupShape[:3]) / arrPoints.max(axis=0)

        arrBlending = np.zeros((n_sources,) + self.tupShape[:3])
        for i in range(n_sources):
            arrCluster = arrPoints[arrLabels == i]
            arrBlending[i,] = _distance_transform(self.tupShape, arrCluster)

        arrBlendingNorm = arrBlending / arrBlending.sum(axis=0)
        arrBlendingNorm = np.tile(arrBlendingNorm[:, :, :, :, np.newaxis], [1, 1, 1, 1, 3])

        if seed:
            np.random.seed(seed)
        arrSources = np.random.choice(np.array(self.warp_sources), (n_sources,), replace=True)
        self.arrWarpSources = arrSources
        self.arrBlendingNormed = arrBlendingNorm
        self.arrWarpBlended = self._combine_warps(arrSources, arrBlendingNorm)

        if verbose:
            print('Generated in {:.03f} s'.format(timeit.default_timer() - t0))

    def augment(self, output_path):
        """Perform augmentation by applying the blended warp.

        Args:
            output_path (str): output file path (.nii*)

        Raises:
            RuntimeError: generate_warp() has not been run or arrWarpBlended has not been assigned.
        """        
        if self.arrWarpBlended is None:
            raise RuntimeError('Run generate_warp first')

        transforms.apply_warp_from_array(self.arrWarpBlended, 
                                        #  arrRefWarp=self.arrWarpBlended,
                                         arrRefWarp=None,
                                         strInFile=self.strRegisteredPath,
                                         strOutPath=output_path, bIsTimeseries=self.bFunc,
                                         strRefFile=self.strTemplate)

    def __del__(self):
        self.tempdir.cleanup()


    def save_warp(self, save_path):
        """Save blended warp to file.

        Args:
            save_path (str): save path (.nii*)
        """     
        if self.arrWarpBlended is None:
            raise RuntimeError('Run generate_warp first')   
        transforms.save_warp(self.arrWarpBlended, arrRefWarp=self.arrWarpBlended, strOutPath=save_path)