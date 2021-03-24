#!/usr/bin/env python
"""
Docstring
______
2/21/20
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'


import os
import re
from nipype import Workflow, MapNode, Node
from nipype.interfaces import (io, fsl, ants, utility, image)
import pandas as pd
from .skullstrip import Robex, make_func_mask_workflow

class T1Registration:
    def __init__(self, input_anat, output_dir):
        self.strOutputDir = output_dir
        self.lsInputs = input_anat

    def get_output_prefix(self, strPath):
        '''
        Get sub-*_ses-* or sub-* prefix from the input file path.
        '''
        strFile = os.path.basename(strPath)
        if 'ses' in strFile:
            match = re.search(r'(sub-.*_ses-.*)_', strFile)
        else:
            match = re.search(r'(sub-.*)_', strFile)
        if match is None:
            raise IOError(strFile, 'is not in a recognizable BIDS format.')
        return match[1]

    def run(self, n_ants_jobs=1, n_pipeline_jobs=1,
            strTemplatePath=None):

        # Use default T1w template if not specified
        if not strTemplatePath:
            strTemplatePath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           'data/mni_icbm152_t1_tal_nlin_sym_09a_brain.nii.gz')
        # If using single-threaded mode, restrict ANTs to single-threaded mode as well
        if n_pipeline_jobs == 1:
            n_ants_jobs = 1
        # Can't have more ANTs threads than total amount of jobs!
        if n_ants_jobs > n_pipeline_jobs:
            raise ValueError('Number of ANTs jobs cannot be higher than the number of total pipeline jobs.')

        if not os.path.exists(self.strOutputDir):
            os.makedirs(self.strOutputDir)

        workflow = Workflow('registration', base_dir=os.path.join(self.strOutputDir))

        # Skull strip with ROBEX
        robex = MapNode(Robex(), name='1_skullstrip', iterfield=['in_file'])
        robex.inputs.in_file = self.lsInputs

        # Register T1w with MNI152 template.
        register = MapNode(ants.RegistrationSynQuick(fixed_image=strTemplatePath, num_threads=n_ants_jobs),
                           name='2_register', iterfield=['moving_image', 'output_prefix'],
                           mem_gb=16, n_procs=n_ants_jobs
                           )
        lsPrefixes = [self.get_output_prefix(s) + '_' for s in self.lsInputs]
        register.inputs.output_prefix = lsPrefixes
        workflow.connect(robex, 'out_file', register, 'moving_image')

        # Softlink output files to output_directory/subject/transformation_type/... 
        datasink = MapNode(io.DataSink(base_directory=self.strOutputDir,
                                       parameterization=False,
                                       remove_dest_dir=True,
                                       infields=['forward_warp_field',
                                                 'inverse_warp_field',
                                                 'out_matrix']),
                           name='datasink',
                           iterfield=['container', 'forward_warp_field',
                                      'inverse_warp_field',
                                      'out_matrix'])
        datasink.inputs.container = [self.get_output_prefix(s) for s in self.lsInputs]
        # Datasink is creating this extra empty folder for no apparent reason, this removes it
        datasink.inputs.regexp_substitutions = [(r'trait_added', r'')]
        workflow.connect([(register, datasink, [('forward_warp_field', 'forward_warp_field'),
                                                ('inverse_warp_field', 'inverse_warp_field'),
                                                ('out_matrix', 'out_matrix')])])

        if n_pipeline_jobs == 1:
            workflow.run()
        else:
            workflow.run(plugin='MultiProc', plugin_args={'n_procs': n_pipeline_jobs})

class EPIRegistration:
    def __init__(self, input_func, output_dir):
        self.strOutputDir = output_dir
        self.lsInputs = input_func

    def get_output_prefix(self, strPath):
        '''
        Get sub-*_ses-* or sub-* prefix from the input file path.
        '''
        strFile = os.path.basename(strPath)
        if 'ses' in strFile:
            match = re.search(r'(sub-.*_ses-.*)_', strFile)
        else:
            match = re.search(r'(sub-.*)_', strFile)
        if match is None:
            raise IOError(strFile, 'is not in a recognizable BIDS format.')
        return match[1]

    def run(self, n_ants_jobs=1, n_pipeline_jobs=1,
            strTemplatePath=None):

        # Use default EPI template if not specified
        if not strTemplatePath:
            strTemplatePath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           'data/tpl-MNI152NLin2009cAsym_res-02_desc-fMRIPrep_boldref_brain.nii.gz')
        # If using single-threaded mode, restrict ANTs to single-threaded mode as well
        if n_pipeline_jobs == 1:
            n_ants_jobs = 1
        # Can't have more ANTs threads than total amount of jobs!
        if n_ants_jobs > n_pipeline_jobs:
            raise ValueError('Number of ANTs jobs cannot be higher than the number of total pipeline jobs.')

        if not os.path.exists(self.strOutputDir):
            os.makedirs(self.strOutputDir)

        workflow = Workflow('registration', base_dir=os.path.join(self.strOutputDir))

        # Compute mean image
        meanimage = MapNode(fsl.MeanImage(dimension='T', output_type='NIFTI_GZ'), name='1_mean', iterfield=['in_file'])
        meanimage.inputs.in_file = self.lsInputs

        # Skull strip
        funcstrip = make_func_mask_workflow(base_dir=self.strOutputDir)
        workflow.connect(meanimage, 'out_file', funcstrip, 'inputnode.mean_file')

        # Register T1w with EPI template.
        register = MapNode(ants.RegistrationSynQuick(fixed_image=strTemplatePath, num_threads=n_ants_jobs),
                           name='2_register', iterfield=['moving_image', 'output_prefix'],
                           mem_gb=16, n_procs=n_ants_jobs
                           )
        lsPrefixes = [self.get_output_prefix(s) + '_' for s in self.lsInputs]
        register.inputs.output_prefix = lsPrefixes
        workflow.connect(funcstrip, 'outputnode.masked_file', register, 'moving_image')

        # Softlink output files to output_directory/subject/transformation_type/...
        datasink = MapNode(io.DataSink(base_directory=self.strOutputDir,
                                       parameterization=False,
                                       remove_dest_dir=True,
                                       infields=['forward_warp_field',
                                                 'inverse_warp_field',
                                                 'out_matrix']),
                           name='datasink',
                           iterfield=['container', 'forward_warp_field',
                                      'inverse_warp_field',
                                      'out_matrix'])
        datasink.inputs.container = [self.get_output_prefix(s) for s in self.lsInputs]
        # Datasink is creating this extra empty folder for no apparent reason, this removes it
        datasink.inputs.regexp_substitutions = [(r'trait_added', r'')]
        workflow.connect([(register, datasink, [('forward_warp_field', 'forward_warp_field'),
                                                ('inverse_warp_field', 'inverse_warp_field'),
                                                ('out_matrix', 'out_matrix')])])

        if n_pipeline_jobs == 1:
            workflow.run()
        else:
            workflow.run(plugin='MultiProc', plugin_args={'n_procs': n_pipeline_jobs})
