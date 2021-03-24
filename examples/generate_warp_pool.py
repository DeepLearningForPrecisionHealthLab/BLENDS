#!/usr/bin/env python
"""
Generate registrations between PPMI T1w images and the MNI template
______
2/21/20
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

# Path to PPMI dataset in BIDS format
DATA_PATH = '/project/bioinformatics/DLLab/STUDIES/PPMI_20191017/Source'
# Output directory to store computed warps
POOL_DIR = '/work/bioinformatics/s169685/augmentation/warps/ppmi/'
# Path to T1-weighted MNI template
TEMPLATE = os.path.abspath('../data/mni_icbm152_t1_tal_nlin_sym_09a_brain.nii.gz')

ANTS_JOBS = 8
PIPELINE_JOBS = 48

import sys, os
import re
import glob
import numpy as np
import pandas as pd
import bids
import transforms3d
import SimpleITK as sitk
from nipype.pipeline import MapNode, Workflow
from nipype.interfaces import ants
from augmentation import registration

ppmi = bids.BIDSLayout(DATA_PATH, validate=False)

def remove_repeats(lsScans):
    # Remove repeated scans for the same subject
    lsOutScans = []
    lsSubjects = []
    for scan in lsScans:
        if scan.entities['subject'] not in lsSubjects:
            lsSubjects += [scan.entities['subject']]
            lsOutScans += [scan.path]
    return lsOutScans

def remove_rigidbody(strAffinePath):
    # Isolate and remove rigid body components (rotation and translation) from
    # the affine matrix, since those transformations likely introduce minimal
    # useful variation into the image.
    affine = sitk.ReadTransform(strAffinePath)
    arrAffine = np.array(affine.GetParameters())[:9].reshape((3, 3))

    arrAffine4 = np.zeros((4, 4))
    arrAffine4[:3, :3] = arrAffine
    arrAffine4[3, 3] = 1
    _, arrRot, arrScale, arrShear = transforms3d.affines.decompose(arrAffine4)

    arrShearMat = np.array([[1, arrShear[0], arrShear[1]],
                            [0, 1, arrShear[2]],
                            [0, 0, 1]])

    arrScaleMat = np.diag(arrScale)
    arrAffineNew = np.dot(arrScaleMat, arrShearMat)

    arrVector = np.concatenate((arrAffineNew.flatten(), np.zeros((3, ))))
    affine.SetParameters(arrVector)

    strOutPath = os.path.join(os.path.dirname(strAffinePath), 'shearing_scaling.mat')
    sitk.WriteTransform(affine, strOutPath)
    return strOutPath

# Find T1-weighted images in the data directory
lsT1 = ppmi.get(suffix='T1w', extensions=['nii', 'nii.gz'])
lsT1Paths = remove_repeats(lsT1)

os.makedirs(POOL_DIR, exist_ok=True)
register = registration.T1Registration(lsT1Paths, POOL_DIR)
register.run(ANTS_JOBS, PIPELINE_JOBS)

# Since ANTs produces affine and nonlinear transform files, we now need to
# compose them into a combined warp file
lsWarpDirs = glob.glob(os.path.join(POOL_DIR, 'sub*'))
lsWarpDirs.sort()

# Find the affine and nonlinear transform files and create output paths for the
# composite transform
dictWarps = {'transforms': [], 'outpath': []}
for strWarpDir in lsWarpDirs:
    strAffinePath = glob.glob(os.path.join(strWarpDir, 'out_matrix', '*.mat'))[0]
    # Remove rigid body components (translation and rotation) which don't
    # contribute meaningful variation
    strAffinePath = remove_rigidbody(strAffinePath)
    # We use the inverse warp field, which contains the nonlinear transformation from MNI->subject
    strNonlinearPath = glob.glob(os.path.join(strWarpDir, 'inverse_warp_field', '*.nii.gz'))[0]
    dictWarps['transforms'].append([strNonlinearPath, strAffinePath])
    dictWarps['outpath'].append(os.path.join(strWarpDir, 'composite_to_mni.nii.gz'))

# Use ANTs ApplyTransforms to compose the transforms
antstool = MapNode(ants.ApplyTransforms(input_image=TEMPLATE,
                                        reference_image=TEMPLATE,
                                        interpolation='BSpline',
                                        invert_transform_flags=[False, True],
                                        print_out_composite_warp_file=True),
                   name='applytransforms',
                   iterfield=['output_image', 'transforms'])
antstool.inputs.output_image = dictWarps['outpath']
antstool.inputs.transforms = dictWarps['transforms']

# Create and run nipype workflow
wf = Workflow('composite_transforms')
wf.add_nodes([antstool])
wf.run(plugin='MultiProc', plugin_args={'n_procs': PIPELINE_JOBS})
