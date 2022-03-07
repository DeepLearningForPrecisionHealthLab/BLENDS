#!/usr/bin/env python
"""
Nipype interface for the ROBEX skullstripping tool. 

Copyright (c) 2022 The University of Texas Southwestern Medical Center.
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

from nipype import Workflow, Node, MapNode
from nipype.interfaces import (io, fsl, ants, utility, image, c3, afni)
from nipype.interfaces.base import (TraitedSpec, CommandLineInputSpec, CommandLine, File, traits)

class RobexInputSpec(CommandLineInputSpec):
    in_file = File(desc="Input volume", exists=True,
                   mandatory=True, position=0, argstr="%s")
    out_file = File(desc="Output volume", position=1, argstr="%s", name_source=['in_file'],
                    hash_files=False, name_template='%s_desc-brain', keep_extension=True)
    out_mask = File(desc="Output mask", position=2, argstr="%s", name_source=['in_file'],
                    hash_files=False, name_template='%s_desc-brain_mask', keep_extension=True)
    seed = traits.Int(desc="seed for random number generator", position=3, argstr="%i")


class RobexOutputSpec(TraitedSpec):
    out_file = File(desc="Output volume", exists=True)
    out_mask = File(desc="Output mask", exists=True)


class Robex(CommandLine):
    """
    Interface for running ROBEX brain extraction. The location of runROBEX.sh in the installation path should be in
    the PATH environmental variable.
    """
    input_spec = RobexInputSpec
    output_spec = RobexOutputSpec
    _cmd = 'runROBEX.sh'

def make_func_mask_workflow(name='funcmask', base_dir=None):
    brainmask = Workflow(name=name, base_dir=base_dir)

    inputnode = Node(utility.IdentityInterface(fields=['mean_file']), name='inputnode')
    outputnode = Node(utility.IdentityInterface(fields=['masked_file', 'mask']),
                      name='outputnode')
    skullstrip1 = MapNode(fsl.BET(frac=0.2, mask=True, output_type='NIFTI_GZ'), name='skullstrip_first_pass',
                          iterfield=['in_file'])
    brainmask.connect(inputnode, 'mean_file', skullstrip1, 'in_file')
    skullstrip2 = MapNode(afni.Automask(dilate=1, outputtype='NIFTI_GZ'), name='skullstrip_second_pass',
                          iterfield=['in_file'])
    brainmask.connect(skullstrip1, 'out_file', skullstrip2, 'in_file')
    combine_masks = MapNode(fsl.BinaryMaths(operation='mul'), name='combine_masks', iterfield=['in_file',
                                                                                               'operand_file'])
    brainmask.connect(skullstrip1, 'mask_file', combine_masks, 'in_file')
    brainmask.connect(skullstrip2, 'out_file', combine_masks, 'operand_file')
    apply_mask = MapNode(fsl.ApplyMask(), name='apply_mask', iterfield=['in_file', 'mask_file'])
    brainmask.connect(inputnode, 'mean_file', apply_mask, 'in_file')
    brainmask.connect(combine_masks, 'out_file', apply_mask, 'mask_file')

    brainmask.connect(apply_mask, 'out_file', outputnode, 'masked_file')
    brainmask.connect(combine_masks, 'out_file', outputnode, 'mask')

    return brainmask