# HCPH Template: A high definition anatomical brain template of one individual healthy subject

We propose a high-definition template of a single healthy brain built using multimodal registration. 35 T1-weighted (T1w) and 35 T2-weighted (T2w) anatomical brain images of one individual healthy human male (aged 40) were retrieved from the Human Connectome Phantom (HCPh) dataset, an ongoing Stage 1 Registered Report.

# Installation

## Python dependencies

The python code is using a set of packages (*Nilearn*, *Nibabel*, *Nitransforms*, *Scikit-Image*, etc).
Use the `requirements.txt` file to install all dependencies at once:
```shell
pip install -r requirements.txt
```

## External dependencies

To run the code you will need some external routines included in open-source packages:
- `N4BiasFieldCorrection`, `DenoiseImage` and `antsMultivariateTemplateConstruction2.sh` from the Advanced Normalization Tools [*ANTs*](https://github.com/ANTsX/ANTs).
- `mri_synthstrip` from [*FreeSurfer*](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/).
Those specific methods should be accessible from you command line for the pipeline to work.

# Running the code

This code has been developped to run on a high power computing (HPC) unit using the SLURM job manager.
It has now been tested and is compatible to run locally.
Before running, make sure your anatomical data (T1w and T2w images) are stored in a BIDS dataset.

## Scripts summary

The processing pipeline uses several scripts. Here's a summary of the steps in each script:

- `create_input_dir.sh`
    - Creates a directory with T1w and T2w images
- `N4-Correction-fortemplate.sh`
    - Skullstripping and computation of the brain mask,
    - INU correction with `N4BiasFieldCorrection`,
    - Image denoising with `DenoiseImage`.
- `histomatch.py`
    - Normalizes the intensity of the images by matching the histogram of each image to the one of a reference (usually the first image).
- `align_with_t1w.sh`
    - Align images of all modalities (different than T1w) with the corresponding T1w.
- `antsMultivariateTemplateConstruction2.sh`
    - Creates the initial template by computing the (affine) transforms between each individual image (moving images) and a reference (fixed, usually the first image).
    - We implemented a modified version (`antsMultivariateTemplateConstruction2-mod.sh`) to bypass the `Walltime` parameter (only applicable for *SLURM* environments).
- `interpolation.py`
    - Creates the final template in the specified resolution by weighting the interpolation by the accuracy of the alignement between voxel centers of the fixed image projected onto the space of the moving image and the center of the closest voxel in the moving space.
- *Optional*: Laplacian sharpening using `ImageMath`:
    - The image can be sharpen using the following command:
    ``` shell
    ImageMath 3 path_to_save_sharp_template Sharpen path_to_template 1
    ```

## Step-by-step run

### 1. Start by adapting the `create_input_dir.sh` script to your system:
    
- You should modify the `sub` and `template_name` variables to match your environments.

- If your anatomical images does not have the `desc-undistorted` tag, replaces the definition of `file_list` with a tag unique to your file names:
    
    ``` shell
    files_list=$( ls $sub_dir/ses-0*anat/*UNIQUE_TAG*.nii.gz )
    ```

Create the input directory:
``` shell
sh create_input_dir.sh ~/data/datasets/hcph-dataset ~/data/hcph-template
```

### 2. **IMPORTANT:** Check that the `templateInput.csv` file matches the T1w and T2w images in the output directory.

### 3. Create a `allImages.txt` file with all anatomical files to be processed:

```shell
ls /path/to/data/*.nii.gz >> allImages.txt
```

### 4. Run the bias-field correction and denoising process

Script details are in `N4-correction-local.sh` or `N4-correction-SLURM.sh` if running on a HPC.

- **IMPORTANT:** Check the quality of some images at each step.
    It is crucial to validate the bias field correction (or correction for Intensity Non-Uniformities (INU)) as any reccurrent non-uniformity will be visible on the template.
    You may do it by looking at the `*corr.nii.gz` images and making sure there are no area that are particularly bright/dark.
    If needed, adjust the parameters of the bias field correction by update the following line in `N4-correction-local.sh`:

    ``` shell
    N4BiasFieldCorrection -d 3 -b [80] -i $imFile -o [$n4corrt1,$n4corrbf]
    ```

To start the pipeline, run
``` shell
sh N4-correction-local.sh ~/data/hcph-template ~/data/hcph-template/allImages.txt
```

**HPC**: Adapt and run the pipeline script using `sbatch N4-correction-pipeline.sbatch`.

### 5. Then, run the `histomatch.py` script to normalize images (see `python histomatch.py -h`).
    
``` shell
python histomatch.py -o ~/data/hcph-template/derivatives/histomatch -m T1w -p your_file_pattern --mask brainmask ~/data/hcph-template/derivatives/n4-corrected
```

### 6. Create another `allImages.txt` file with all anatomical files to be further processed:
    
```shell
cd ~/data/hcph-template/derivatives/histomatch
ls ./*.nii.gz >> allImages.txt
```

### 7. Align the modalities (other than T1w) to the T1w images using `align_to_t1w.sh`.
   
``` shell
sh align_with_t1w-local.sh ~/data/hcph-template/derivatives/histomatch ~/data/hcph-template/derivatives/histomatch/allImages.txt
```

**HPC**: Adapt and run the pipeline script using `sbatch align_to_t1w.sbatch`.

### 8. Create the input file to build the template

`antsMultivariateTemplateConstruction2.sh` requires a specific input that we create using `hcph-template_gen_in_file.sh`.
    
``` shell
sh hcph-template_gen_in_file.sh ~/data/hcph-template/derivatives/allInRef
```

### 9. Finally, build the template.

Check that the `templateInput.csv` file has all desired images, then run:
``` shell
sh hcph-template_build_template-local.sh ~/data/hcph-template/derivatives/allInRef
```
**HPC**: Adapt and run the pipeline script using `sbatch hcph-template_build_template_mni_sl2.sbatch`.

### 10. Inspect the quality of the template

Have a look at the `A_tpl_template0.nii.gz` and `A_tpl_template1.nii.gz`.

### 11. *Optional:* Refine the template

You may run the script to build the template with the latest template as input (e.g. `-z my_initial_template.nii.gz`, see `antsMultivariateTemplateConstruction2.sh` documentation for examples)

### 12. *Optional:* Register the latest template to another reference space (e.g. MNI).

``` shell
sh template_to_reference.sh ~/data/hcph-template/derivatives/diswe-interp/my_template.nii.gz ~/path_to_mni/mni_template.nii.gz MNI04mm Affine
```

### 13. Run the interpolation process

See `python interpolation.py -h` for all parameters.
``` shell
python interpolation.py -o ~/data/hcph-template/derivatives/diswe-interp -r 0.4 --transform-dir ANTs_iteration_3 -b 500 --exclude-ses ses-pilot019 ses-pilot021 -j 10 --pre-transform T2w-to-T1w --post-transform to_MNI04mmAffine --use-mni -m T2w ~/data/hcph-template/derivatives/histomatch ~/data/hcph-template/derivatives/allInRef
```

Note that in the above call, we are using most of the optional parameters that are:
- `--exclude-ses`:
    Exclude specific session (e.g. corrupted data) in the interpolation.

- `--pre-transform`:
    Add intermediate transformation between the session space (T1w) and the image space (usually T2w or other modalities, one for each image).
    Note that the transform should have been already computed before (see step 7) and files should be saved in the same directory as the transforms from *ANTs* (last argument).

- `--post-transform`:
    Add intermediate transformation etween the template space and a final resampling space (e.g. MNI space after Affine or Rigid transformation).
    The transform file should be computed separately (see below) and saved in the same folder as the transforms from *ANTs*.

- `--use-mni`:
    Will consider the MNI template as reference grid in the desired resolution (see [Nilearn's MNI template](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.load_mni152_template.html#nilearn.datasets.load_mni152_template)).