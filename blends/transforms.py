#!/usr/bin/env python
"""
Save and apply ITK warp fields

Copyright (c) 2022 The University of Texas Southwestern Medical Center.
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

import os
import numpy as np
from skimage import exposure
import SimpleITK as sitk
import nilearn as nil
from nipype.interfaces import ants, image
from nipype.pipeline import Node, Workflow
import tempfile

strTemplatePath = '/project/bioinformatics/DLLab/Kevin/Augmentation_fMRI/data/mni_icbm152_t1_tal_nlin_sym_09a_brain.nii.gz'
strAntsInstallPath = '/project/bioinformatics/DLLab/softwares/ants2.3.1_build/bin'


def save_warp(arrWarp, strOutPath, arrRefWarp=None, nVoxSize=(3,3,3)):
    '''
    Convert a numpy array warp field into an ITK displacement field file
    :param arrWarp: 4-dimensional warp field
    :type arrWarp: numpy.ndarray
    :param strOutPath: path to save ITK file
    :type strOutPath: str
    :param arrRefWarp: a reference warp field (e.g. a real warp) used to perform histogram matching. If none is
    specified, the input warp field will be min-max scaled to -7..7
    :type arrRefWarp: numpy.ndarray
    :param nVoxSize: voxel size in mm. default (3, 3, 3)
    :type nVoxSize: tuple
    :return:
    :rtype:
    '''
    arrAffine = np.array([[nVoxSize[0], 0, 0, -98],
                          [0, nVoxSize[1], 0, -134],
                          [0, 0, nVoxSize[2], -72],
                          [0, 0, 0, 1]])

    if isinstance(arrRefWarp, np.ndarray):
        # Match histogram with reference image
        arrWarpImg = exposure.match_histograms(arrWarp, arrRefWarp)
    else:
        # Roughly min-max scale to -7..7
        arrWarpImg = arrWarp.copy()
        arrWarpImg *= (7 / np.abs(arrWarpImg).max())
    #
    # arrWarpImg = arrWarpImg.squeeze()
    # imgWarp = nil.image.new_img_like(ref_niimg=strTemplatePath, data=arrWarpImg, affine=arrAffine)
    # imgWarp = nil.image.resample_to_img(imgWarp, strTemplatePath, interpolation='continuous')
    #
    # arrWarp = imgWarp.get_data().astype(np.float64)
    imgSitk = sitk.GetImageFromArray(np.swapaxes(arrWarp, 0, 2), isVector=True)
    imgTemp = sitk.ReadImage(strTemplatePath)
    imgSitk.SetOrigin(imgTemp.GetOrigin())
    imgSitk.SetDirection(imgTemp.GetDirection())
    imgSitk.SetSpacing(imgTemp.GetSpacing())
    sitk.WriteImage(imgSitk, strOutPath)

def apply_warp_ants(strWarpPath, strInFile=None, strOutPath=None, bIsTimeseries=False, strRefFile=None,
                    strAntsPath=None):
    '''
    Apply a warp field (in ITK displacement field format) to an image
    :param strWarpPath: path to a warp field (.nii.gz)
    :type strWarpPath: str
    :param strInFile: path to input image. Defaults to MNI T1 template.
    :type strInFile: str
    :param strOutPath: output path for warped image. Will create a temp path if not specified.
    :type strOutPath: str
    :param strAntsPath: path to ANTS binary installation
    :type strAntsPath: str
    :return: if strOutPath is not specified, returns the warped image as a Nibabel Nifti1 object
    :rtype:
    '''
    if strInFile is None:
        strInFile = strTemplatePath
    if strOutPath is None:
        bTempDir = True
        tempdir = tempfile.TemporaryDirectory()
        strOutPath = os.path.join(str(tempdir.name), 'warped.nii.gz')
    else:
        bTempDir = False
    if strAntsPath == None:
        strAntsPath = strAntsInstallPath

    nImType = 3 if bIsTimeseries else 0
    if strRefFile is None:
        strRefFile = strInFile
    transform = ants.ApplyTransforms(input_image=strInFile, reference_image=strRefFile, transforms=strWarpPath,
                                     input_image_type=nImType,
                                     dimension=3, interpolation='BSpline', output_image=strOutPath,
                                     environ={'PATH': '$PATH:{}'.format(strAntsPath)})
    out = transform.run()

    # os.system('{}/antsApplyTransforms -i {} -r {} -o {} -t {} -n BSpline --input-image-type {}'.format(strAntsPath,
    #                                                                                              strInFile, strInFile,
    #                                                                                         strOutPath, strWarpPath,
    #                                                                                                    nImType))

    if bTempDir:
        img = nil.image.load_img(strOutPath)
        tempdir.cleanup()
        return img

def apply_warp_from_array(arrWarp, arrRefWarp, strInFile=None, strOutPath=None, strRefFile=None, nVoxSize=(3,3,3),
                          bIsTimeseries=False):
    '''
    Apply a warp field (in a numpy array) to an image
    :param arrWarp: 4-dimensional warp field
    :type arrWarp: numpy.ndarray
    :param arrRefWarp: a reference warp field (e.g. a real warp) used to perform histogram matching. If none is
    specified, the input warp field will be min-max scaled to -7..7
    :type arrRefWarp: numpy.ndarray
    :param strInFile: path to input image. Defaults to MNI T1 template.
    :type strInFile: str
    :param strOutPath: output path for warped image. Will create a temp path if not specified.
    :type strOutPath: str
    :param nVoxSize: voxel size in mm. default (3, 3, 3)
    :type nVoxSize: tuple
    :return: if strOutPath is not specified, returns the warped image as a Nibabel Nifti1 object. Otherwise,
    returns None
    :rtype:
    '''
    with tempfile.TemporaryDirectory() as tempdir:
        strWarpPath = os.path.join(tempdir, 'warp.nii.gz')
        save_warp(arrWarp, strWarpPath, arrRefWarp, nVoxSize)
        img = apply_warp_ants(strWarpPath, strInFile, strOutPath, strRefFile=strRefFile, bIsTimeseries=bIsTimeseries)
    return img