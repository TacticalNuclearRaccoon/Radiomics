
# Multiclass segmentation of brain tumors

## Image data descriptions
All BraTS multimodal scans are available as NIfTI files (.nii.gz) -> commonly used medical imaging format to store brain imageing data obtained using MRI and describe different MRI settings

* T1: T1-weighted, native image, sagittal or axial 2D acquisitions, with 1–6 mm slice thickness.
* T1c: T1-weighted, contrast-enhanced (Gadolinium) image, with 3D acquisition and 1 mm isotropic voxel size for most patients.
* T2: T2-weighted image, axial 2D acquisition, with 2–6 mm slice thickness.
* FLAIR: T2-weighted FLAIR image, axial, coronal, or sagittal 2D acquisitions, 2–6 mm slice thickness.

Data were acquired with different clinical protocols and various scanners from multiple (n=19) institutions.

All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise: <br> 
* the GD-enhancing tumor (ET — label 4), 
* the peritumoral edema (ED — label 2), 
* and the necrotic and non-enhancing tumor core (NCR/NET — label 1), 

as described both in the BraTS 2012-2013 TMI [paper](https://ieeexplore.ieee.org/document/6975210) and in the latest BraTS summarizing paper. <br>

In this notebook we will use segmentation labels 0, 1, 2, 3 instead of (0, 1, 2, 4) because we fail to understand the logic of the original paper.   

The provided data are distributed after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution and skull-stripped.

## Problem statement

**Segmentation of gliomas in pre-operative MRI scans:**

Task: Each pixel on image must be labeled: <br>

* If the pixel is not on a tumor region it is background.
    - 0: background
* If the pixel is part of the tumor area, it can belong to the following subclasses: <br>
    - 1: necrotic (non-enhancing) tumor core 
    - 2: edema
    - 3: enhancing tumor
