#!/usr/bin/env python
"""
Classes for performing a simple augmentation of 3D/4D images via random affine transformations. 

Copyright (c) 2022 The University of Texas Southwestern Medical Center.
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

import os
import re
import numpy as np
import pandas as pd
from nipype import Workflow, MapNode, Node
from nipype.interfaces import utility, fsl
from . import skullstrip

class AffineAugmenter:
    def __init__(self, input_files, output_dir, n_augment,
                 lr_trans_range=np.arange(-10, 10, 1),
                 ap_trans_range=np.arange(-5, 15, 1),
                 si_trans_range=np.arange(-3, 3, 1),
                 lr_rot_range=np.arange(-20, 20, 1),
                 ap_rot_range=np.arange(-20, 20, 1),
                 si_rot_range=np.arange(-20, 20, 1),
                 scale_range=np.arange(0.9, 1.1, 0.05),
                 seed=None):
        """Augmentation through randomly generated affine transformations.

        Initialize with desired transformation ranges, then call .run(). Input
        images must be organized in a BIDS-style structure.

        Requires FSL to be installed and on the system PATH.

        Args:
            input_files (list): list of original images with BIDS-compliant filename and directory structure
            output_dir (str): path to output location
            n_augment (int): number of times to augment each original image
            lr_trans_range (array, optional): left-right translation choices. Defaults to np.arange(-10, 10, 1).
            ap_trans_range (array, optional): anterior-posterior translation choices. Defaults to np.arange(-5, 15, 1).
            si_trans_range (array, optional): superior-inferior translation choices. Defaults to np.arange(-3, 3, 1).
            lr_rot_range (array, optional): rotation on left-right axis in degrees. Defaults to np.arange(-20, 20, 1).
            ap_rot_range (array, optional): rotation on anterior-posterior axis in degrees. Defaults to np.arange(-20, 20, 1).
            si_rot_range (array, optional): rotation on superior-inferior axis in degrees. Defaults to np.arange(-20, 20, 1).
            scale_range (array, optional): relative isotropic scaling. Defaults to np.arange(0.9, 1.1, 0.05).
            seed (int, optional): random seed. Defaults to None.
        """        
        self.strOutputDir = os.path.abspath(output_dir)
        self.dfConfig = pd.DataFrame()
        nImages = len(input_files)
        lsFunc = []
        lsAnat = []
        lsOutputFunc = []
        lsOutputAnat = []
        for tup in input_files:
            lsFunc += [tup[0]] * n_augment
            lsAnat += [tup[1]] * n_augment
            lsOutputFunc += self._generate_output_name(tup[0], n_augment)
            lsOutputAnat += self._generate_output_name(tup[1], n_augment)
        self.dfConfig['func'] = lsFunc
        self.dfConfig['anat'] = lsAnat
        self.dfConfig['output_func'] = lsOutputFunc
        self.dfConfig['output_anat'] = lsOutputAnat
        nAugmentedImages = nImages * n_augment
        for strParam in ['lr_trans', 'ap_trans', 'si_trans', 'lr_rot', 'ap_rot', 'si_rot', 'scale']:
            if seed:
                np.random.seed(seed)
            self.dfConfig[strParam] = np.random.choice(locals()[strParam + '_range'], nAugmentedImages, True)

    def _generate_output_name(self, in_file, n_augment):
        match = re.search(r'sub-([a-zA-Z0-9]+)_ses-([a-zA-Z0-9]+)_(.*)', in_file)
        strSub = match.group(1)
        strSes = match.group(2)
        strSuffix = match.group(3)
        strType = 'anat' if 'anat' in in_file else 'func'
        for n in range(n_augment):
            strDir = os.path.join(self.strOutputDir, 'sub-{sub}aug{n:02d}/ses-{ses}/{type}').format(sub=strSub, n=n,
                                                                                                   ses=strSes,
                                                                                                   type=strType)
            if not os.path.exists(strDir):
                os.makedirs(strDir)
            strOutTemp = os.path.join(strDir, 'sub-{sub}aug{n:02d}_ses-{ses}_{suffix}').format(sub=strSub, n=n,
                                                                                               ses=strSes, suffix=strSuffix)
            yield strOutTemp

    def _create_fsl_mats(self):
        if not os.path.exists(os.path.join(self.strOutputDir, 'mats')):
            os.makedirs(os.path.join(self.strOutputDir, 'mats'))
        for n, row in self.dfConfig.iterrows():
            # Create translation and scaling matrix
            arrTrans = np.array([[row['scale'], 0, 0, row['lr_trans']],
                                 [0, row['scale'], 0, row['ap_trans']],
                                 [0, 0, row['scale'], row['si_trans']],
                                 [0, 0, 0, 1]])
            # Create the 3 rotation matrices
            fRotLR = row['lr_rot'] * np.pi / 180
            fRotAP = row['ap_rot'] * np.pi / 180
            fRotSI = row['si_rot'] * np.pi / 180
            arrRotLR = np.array([[1, 0, 0, 0],
                                 [0, np.cos(fRotLR), -np.sin(fRotLR), 0],
                                 [0, np.sin(fRotLR), np.cos(fRotLR), 0],
                                 [0, 0, 0, 1]])
            arrRotAP = np.array([[np.cos(fRotAP), 0, -np.sin(fRotAP), 0],
                                 [0, 1, 0, 0],
                                 [np.sin(fRotAP), 0, np.cos(fRotLR), 0],
                                 [0, 0, 0, 1]])
            arrRotSI = np.array([[np.cos(fRotSI), -np.sin(fRotSI), 0, 0],
                                 [np.sin(fRotSI), np.cos(fRotSI), 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
            arrCombined = arrRotSI * arrRotAP * arrRotLR * arrTrans
            # write to file
            strPath = os.path.basename(row['output_func']).split('.')[0]
            strSavePath = os.path.join(self.strOutputDir, 'mats', strPath + '.mat')
            np.savetxt(strSavePath, arrCombined)
            self.dfConfig.at[n, 'mat_path'] = strSavePath

    def run(self, n_pipeline_jobs=1):
        """Run augmentations

        Args:
            n_pipeline_jobs (int, optional): Number of parallel processing jobs. Defaults to 1.
        """        
        self._create_fsl_mats()
        if not os.path.exists(self.strOutputDir):
            os.makedirs(self.strOutputDir)
        strJobListPath = os.path.join(self.strOutputDir, 'joblist.csv')
        self.dfConfig.to_csv(strJobListPath)

        datanode = Node(utility.csv.CSVReader(in_file=os.path.abspath(strJobListPath), header=True), name='datanode')

        augment = Workflow('augmentation_affine', base_dir=os.path.join(self.strOutputDir, 'working_dir'))

        transformFunc = MapNode(fsl.ApplyXFM(interp='spline', apply_xfm=True), name='transform_func',
                                iterfield=['in_file', 'reference', 'in_matrix_file', 'out_file'])
        augment.connect(datanode, 'func', transformFunc, 'in_file')
        augment.connect(datanode, 'func', transformFunc, 'reference')
        augment.connect(datanode, 'mat_path', transformFunc, 'in_matrix_file')
        augment.connect(datanode, 'output_func', transformFunc, 'out_file')

        transformAnat = MapNode(fsl.ApplyXFM(interp='spline', apply_xfm=True), name='transform_anat',
                                iterfield=['in_file', 'reference', 'in_matrix_file', 'out_file'])
        augment.connect(datanode, 'anat', transformAnat, 'in_file')
        augment.connect(datanode, 'anat', transformAnat, 'reference')
        augment.connect(datanode, 'mat_path', transformAnat, 'in_matrix_file')
        augment.connect(datanode, 'output_anat', transformAnat, 'out_file')

        if n_pipeline_jobs == 1:
            augment.run()
        else:
            augment.run(plugin='MultiProc', plugin_args={'n_procs': n_pipeline_jobs})


class PredefinedAffineAugmenter:
    def __init__(self, input_func, input_anat, affines, output_func, output_anat, output_dir):
        """Apply a predefined set of affine transformations to a list of images.

        Args:
            input_func (list): list of original fMRI images
            input_anat (list): list of original structural MRI, corresponding to each original fMRI
            affines (list): list of affine matrices to apply, in FSL .mat format
            output_func (list): list of output paths for each transformed fMRI
            output_anat (list): list of output paths for each transformed structural MRI
            output_dir (str): path to store intermediate processing files
        """        
        self.strOutputDir = os.path.abspath(output_dir)
        self.dfConfig = pd.DataFrame()
        self.dfConfig['func'] = input_func
        self.dfConfig['anat'] = input_anat
        self.dfConfig['output_func'] = output_func
        self.dfConfig['output_anat'] = output_anat
        self.dfConfig['affine'] = affines

    def run(self, n_pipeline_jobs=1):
        """Perform transformations.

        Args:
            n_pipeline_jobs (int, optional): number of parallel processing jobs. Defaults to 1.
        """        
        if not os.path.exists(self.strOutputDir):
            os.makedirs(self.strOutputDir)
        strJobListPath = os.path.join(self.strOutputDir, 'joblist.csv')
        self.dfConfig.to_csv(strJobListPath)

        datanode = Node(utility.csv.CSVReader(in_file=os.path.abspath(strJobListPath), header=True),
                        name='datanode')

        augment = Workflow('augmentation_affinereg', base_dir=os.path.join(self.strOutputDir, 'working_dir'))

        transformFunc = MapNode(fsl.ApplyXFM(interp='spline', apply_xfm=True), name='transform_func',
                                iterfield=['in_file', 'reference', 'in_matrix_file', 'out_file'])
        augment.connect(datanode, 'func', transformFunc, 'in_file')
        augment.connect(datanode, 'func', transformFunc, 'reference')
        augment.connect(datanode, 'affine', transformFunc, 'in_matrix_file')
        augment.connect(datanode, 'output_func', transformFunc, 'out_file')

        transformAnat = MapNode(fsl.ApplyXFM(interp='spline', apply_xfm=True), name='transform_anat',
                                iterfield=['in_file', 'reference', 'in_matrix_file', 'out_file'])
        augment.connect(datanode, 'anat', transformAnat, 'in_file')
        augment.connect(datanode, 'anat', transformAnat, 'reference')
        augment.connect(datanode, 'affine', transformAnat, 'in_matrix_file')
        augment.connect(datanode, 'output_anat', transformAnat, 'out_file')

        if n_pipeline_jobs == 1:
            augment.run()
        else:
            augment.run(plugin='MultiProc', plugin_args={'n_procs': n_pipeline_jobs})