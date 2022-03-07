#!/usr/bin/env python
"""
Tools for visualizing warps and applying them to images

Copyright (c) 2022 The University of Texas Southwestern Medical Center.
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

from matplotlib import pyplot as plt
import numpy as np

def plot_warp(arrWarps):
    if len(arrWarps.shape) == 4:
        warps = arrWarps[np.newaxis,]
    else:
        warps = arrWarps
    nWarps = warps.shape[0]
    fig, ax = plt.subplots(nWarps, 3)
    if len(ax.shape) < 2:
        ax = np.reshape(ax, (1, -1))
    for i in range(nWarps):
        arrWarpSingle = warps[i,].copy()
        # scale to 0..1
        arrWarpSingle -= arrWarpSingle.min()
        arrWarpSingle /= arrWarpSingle.max()

        tupShape = arrWarpSingle.shape
        arrSliceAx = arrWarpSingle[:, :, tupShape[2] // 2, :]
        arrSliceCor = arrWarpSingle[:, tupShape[1] // 2, :, :]
        arrSliceSag = arrWarpSingle[tupShape[0] // 2, :, :, :]
        ax[i, 0].imshow(np.rot90(arrSliceAx))
        ax[i, 1].imshow(np.rot90(arrSliceCor))
        ax[i, 2].imshow(np.rot90(arrSliceSag))
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
        ax[i, 2].axis('off')
    return fig, ax