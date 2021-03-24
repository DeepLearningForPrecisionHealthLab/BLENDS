#!/usr/bin/env python
"""
"Old" augmentation method reported in Nguyen et al. SPIE 2020. Performs
augmentation by directly registering a source brain to a selected target brain.
Due to the multiple registration steps required, this is much slower than
BLENDS.
______
6/19/19
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

import os
from nipype import Workflow, MapNode, Node
from nipype.interfaces import io, ants, utility, fsl, image
import pandas as pd
from .skullstrip import Robex, make_func_mask_workflow

class AnatAugmenter:
    def __init__(self, output_dir):
        self.strOutputDir = output_dir
        self.dfConfig = pd.DataFrame(columns=['func', 'anat', 'target_func', 'target_anat', 'output_func',
                                              'output_anat'])

    def add_job(self, source_func, source_anat, target_func, target_anat, output_func, output_anat):
        for strFile in [source_func, source_anat, target_func, target_anat]:
            if not os.path.exists(strFile):
                raise FileNotFoundError(strFile + ' does not exist')
        output_func = os.path.abspath(output_func)
        output_anat = os.path.abspath(output_anat)
        for strFile in [output_func, output_anat]:
            strDir = os.path.dirname(strFile)
            if not os.path.exists(strDir):
                os.makedirs(strDir)

        self.dfConfig = self.dfConfig.append({'func': source_func, 'anat': source_anat, 'target_func': target_func,
                                  'target_anat': target_anat, 'output_func': output_func, 'output_anat': output_anat},
                                             ignore_index=True)

    def run(self, do_skullstrip=True, n_ants_jobs=1, n_pipeline_jobs=1):
        if n_pipeline_jobs == 1:
            n_ants_jobs = 1

        if not os.path.exists(self.strOutputDir):
            os.makedirs(self.strOutputDir)
        strJobListPath = os.path.join(self.strOutputDir, 'joblist.csv')
        self.dfConfig.to_csv(strJobListPath)

        datanode = Node(utility.csv.CSVReader(in_file=os.path.abspath(strJobListPath), header=True), name='0_datanode')

        augment = Workflow('augmentation', base_dir=os.path.join(self.strOutputDir, 'working_dir'))

        reorientFunc = MapNode(image.Reorient(), name='0_reorient_func', iterfield=['in_file'])
        augment.connect(datanode, 'func', reorientFunc, 'in_file')
        reorientAnat = MapNode(image.Reorient(), name='0_reorient_anat', iterfield=['in_file'])
        augment.connect(datanode, 'anat', reorientAnat, 'in_file')
        reorientTargetFunc = MapNode(image.Reorient(), name='0_reorient_targetfunc', iterfield=['in_file'])
        augment.connect(datanode, 'target_func', reorientTargetFunc, 'in_file')
        reorientTargetAnat = MapNode(image.Reorient(), name='0_reorient_targetanat', iterfield=['in_file'])
        augment.connect(datanode, 'target_anat', reorientTargetAnat, 'in_file')

        meanFunc = MapNode(fsl.MeanImage(), name='1_mean_func', iterfield=['in_file'])
        augment.connect(reorientFunc, 'out_file', meanFunc, 'in_file')
        meanTargetFunc = MapNode(fsl.MeanImage(), name='1_mean_targetfunc', iterfield=['in_file'])
        augment.connect(reorientTargetFunc, 'out_file', meanTargetFunc, 'in_file')

        if do_skullstrip:
            # Skull strip the anatomical images with ROBEX
            skullstripSourceAnat = MapNode(Robex(), name='1_source_anat_skullstrip', iterfield=['in_file'])
            augment.connect(reorientAnat, 'out_file', skullstripSourceAnat, 'in_file')
            skullstripTargetAnat = MapNode(Robex(), name='1_target_anat_skullstrip', iterfield=['in_file'])
            augment.connect(reorientTargetAnat, 'out_file', skullstripTargetAnat, 'in_file')

            # Skull strip the functional image with FSL BET and AFNI Automask
            skullstripSourceFunc = make_func_mask_workflow(base_dir=os.path.join(self.strOutputDir, 'working_dir'))
            augment.connect(meanFunc, 'out_file', skullstripSourceFunc, 'inputnode.mean_file')


        # First, perform a quick registration of the functional skull-stripped mean image to the anatomical
        # skull-stripped image. Use the SynQuick tool which does a rigid->affine->syn registration with some preset params
        func2Anat = MapNode(ants.RegistrationSynQuick(dimension=3, num_threads=n_ants_jobs),
                            name='2_func2Anat', mem_gb=16, n_procs=n_ants_jobs,
                            iterfield=['fixed_image', 'moving_image'])

        if do_skullstrip:
            augment.connect(skullstripSourceAnat, 'out_file', func2Anat, 'fixed_image')
            augment.connect(skullstripSourceFunc, 'outputnode.masked_file', func2Anat, 'moving_image')
        else:
            augment.connect(reorientAnat, 'out_file', func2Anat, 'fixed_image')
            augment.connect(reorientFunc, 'out_file', func2Anat, 'moving_image')

        # Now register the source anatomical image to the target anatomical image. Use a more precise registration for
        # this step. These parameters come from the antsRegistrationSyn script included in ANTS
        anat2Anat = MapNode(ants.Registration(metric=['MI', 'MI', 'CC'],
                                              metric_weight=[1, 1, 1],
                                              transforms=['Rigid', 'Affine', 'SyN'],
                                              smoothing_sigmas=[[3, 2, 1, 0]] * 3,
                                              shrink_factors=[[8, 4, 2, 1]] * 3,
                                              dimension=3,
                                              initial_moving_transform_com=1,
                                              radius_or_number_of_bins=[32, 32, 4],
                                              sampling_strategy=['Regular', 'Regular', None],
                                              sampling_percentage=[0.25, 0.25, None],
                                              use_histogram_matching=True,
                                              collapse_output_transforms=True,
                                              write_composite_transform=True,
                                              transform_parameters=[(0.1,), (0.1,), (0.1, 3, 0)],
                                              number_of_iterations=[[1000, 500, 250, 100],
                                                                    [1000, 500, 250, 100],
                                                                    [100, 70, 50, 20]],
                                              sigma_units=['vox'] * 4,
                                              winsorize_upper_quantile=0.995,
                                              winsorize_lower_quantile=0.005,
                                              num_threads=n_ants_jobs,
                                              verbose=False),
                            name='3_anat2Anat', iterfield=['fixed_image', 'moving_image', 'output_warped_image'],
                            mem_gb=16, n_procs=n_ants_jobs
                            )

        if do_skullstrip:
            augment.connect(skullstripSourceAnat, 'out_file', anat2Anat, 'moving_image')
            augment.connect(skullstripTargetAnat, 'out_file', anat2Anat, 'fixed_image')
        else:
            augment.connect(reorientAnat, 'out_file', anat2Anat, 'moving_image')
            augment.connect(reorientTargetAnat, 'out_file', anat2Anat, 'fixed_image')
        augment.connect(datanode, 'output_anat', anat2Anat, 'output_warped_image')

        # Finally, use the func-to-anat transform, then the anat-to-anat transform on the source functional image
        concat = MapNode(utility.Merge(3), name='4_concat_transforms', iterfield=['in1', 'in2', 'in3'])
        # Ants applies transforms in reverse order. The first transform is the affine func-to-anat
        augment.connect(func2Anat, 'out_matrix', concat, 'in3')
        # then the nonlinear func-to-anat
        augment.connect(func2Anat, 'forward_warp_field', concat, 'in2')
        # and lastly the composite anat-to-anat
        augment.connect(anat2Anat, 'composite_transform', concat, 'in1')

        transform = MapNode(ants.ApplyTransforms(input_image_type=3, interpolation='BSpline', dimension=3,
                                     interpolation_parameters=(5,), num_threads=n_ants_jobs), name='4_apply_transforms',
                            iterfield=['input_image', 'transforms', 'output_image', 'reference_image'], mem_gb=16,
                            n_procs=n_ants_jobs)
        augment.connect(concat, 'out', transform, 'transforms')
        augment.connect(reorientFunc, 'out_file', transform, 'input_image')
        augment.connect(meanTargetFunc, 'out_file', transform, 'reference_image')
        augment.connect(datanode, 'output_func', transform, 'output_image')

        if n_pipeline_jobs == 1:
            augment.run()
        else:
            augment.run(plugin='MultiProc', plugin_args={'n_procs': n_pipeline_jobs})

