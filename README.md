# The BLENDS Method for Data Augmentation of 4-Dimensional Brain Images
Manuscript currently under review.

Brain Library Enrichment through Nonlinear Deformation Synthesis (BLENDS) is a method for data augmentation of brain images, including 3D+time images such as fMRI.

## Dependencies
### Python packages
See the requirements.txt file. 
* numpy >= 1.15.4
* pandas >= 0.25.3
* nipype >= 1.4.2
* nibabel >= 2.4.0.dev0
* tqdm >= 4.49.0
* SimpleITK >= 1.1.0
* nilearn >= 0.6.2
* scikit-image >= 0.16.2
* matplotlib >= 2.2.2
* transforms3d >= 0.3.1
* bids >= 0.0

### Neuroimage processing tools
These tools should be installed and added to the system PATH.
* ANTs >= 2.2.0
* ROBEX >= 1.2
* FSL >= 5.0.10

## Usage
Input images are assumed to be organized in BIDS-compliant directory and filename structure. Tools like [PyBIDS](https://bids-standard.github.io/pybids/) can be used to organize existing neuroimaging datasets into BIDS structure. 

The `blends.blending.WarpBlender` class contains the main functionality for running BLENDS on an fMRI or a pair of fMRI + T1-weighted MRI (sMRI) from the same subject. Running BLENDS involves initializing `WarpBlender` with a directory containing precomputed nonlinear warps and then calling 3 methods. 
Steps: 

0. Create a `WarpBlender` using a directory of precomputed MNI->subject warps (see below).
```
augmenter = WarpBlender(warp_directory)
```
1.  Use `prepare_source()` to skullstrip the input image and register it to the MNI template.
```
augmenter.prepare_source(func_path, registration='func')
```
2.  Use `generate_warp()` to compute the new spatially blended warp. By default, this will blend together 4 of the original warps. 
```
augmenter.generate_warp()
```
3.  Call `augment()`. 
```
augmenter.augment(output_path)
```
See the docstring for `WarpBlender` for more information on method arguments.

### Generating warp pool
An example script for generating a pool of MNI->subject warps is included: [`examples/generate_warp_pool.py`](./examples/generate_warp_pool.py)

### Other functionality
Also included are a class for performing random affine augmentations (`blends.affine.AffineAugmenter`) and an implementation of the EROS SMOTE method (`blends.smote.SmoteEros`). 

## Acknowledgements
The BOLD MNI152 template in `data/` was obtained from the [templateflow](https://github.com/templateflow/templateflow) repository.