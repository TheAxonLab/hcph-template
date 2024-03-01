# HCPH Template: A high definition anatomical brain template of one individual healthy subject

## Installation

### Dependencies

To run the code you will need some external routines included in open-source packages:
- `N4BiasFieldCorrection`, `DenoiseImage` and `antsMultivariateTemplateConstruction2.sh` from the Advanced Normalization Tools [*ANTs*](https://github.com/ANTsX/ANTs).
- `mri_synthstrip` from [*FreeSurfer*](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/).

## Running the code

This code has been developped to run on a high power computing (HPC) unit using the SLURM job manager.
Some tips are added to run the code locally.
Before running, make sure your anatomical data (T1w and T2w images) are stored in a BIDS dataset.

### Running in a *Slurm* environment (HPC)

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
- `antsMultivariateTemplateConstruction2-mod.sh`
    - Creates the initial template by computing the (affine) transforms between each individual image (moving images) and a reference (fixed, usually the first image).
- `interpolation.py`
    - Creates the final template in the specified resolution by weighting the interpolation by the accuracy of the alignement between voxel centers of the fixed image projected onto the space of the moving image and the center of the closest voxel in the moving space.
- *Optional*: Laplacian sharpening using `ImageMath`:
    - The image can be sharpen using the following command:
    ``` shell
    ImageMath 3 path_to_save_sharp_template Sharpen path_to_template 1
    ```

#### Step-by-step run

1. Start by adapting the `create_input_dir.sh` script to your system:
    
    - You should modify the `bids_dir`, `sub`, `save_dir` and `template_name` variables to match your environments.

    - If your anatomical images does not have the `desc-undistorted` tag, replaces the definition of `file_list` with a tag unique to your file names:
    
        ``` shell
        files_list=$( ls $sub_dir/ses-0*anat/*UNIQUE_TAG*.nii.gz )
        ```

    Create the input directory:
    ``` shell
    sh create_input_dir.sh
    ```

2. **IMPORTANT:** Check that the `templateInput.csv` file matches the T1w and T2w images in the output directory.

3. Create a `allImages.txt` file with all anatomical files to be processed:

    ```shell
    ls /path/to/data/*.nii.gz >> allImages.txt
    ```

4. In your *SLURM* environment, run the denoising process (script details are in `N4-Correction-fortemplate.sh`).

    **IMPORTANT:** Check the quality of some images at each step.
    It is crucial to validate the bias field correction (or correction for Intensity Non-Uniformities (INU)) as any reccurrent non-uniformity will be visible on the template.
    You may do it by looking at the `*corr.nii.gz` images and making sure there are no area that are particularly bright/dark.
    If needed, adjust the parameters of the bias field correction by update the following line in `N4-Correction-fortemplate.sh`:
    ``` shell
    N4BiasFieldCorrection -d 3 -b [80] -i $imFile -o [$n4corrt1,$n4corrbf]
    ```
    To start the pipeline, run:
    ``` shell
    sbatch N4-Correction_Pipeline_cl2.sbatch
    ```

5. Then, run the `histomatch.py` script to normalize images.
    ``` shell
    python histomatch.py -o path_to_output -m T2w -p your_file_pattern path_to_n4corrected_data
    ```

6. Align the modalities (other than T1w) to the T1w images using `align_to_t1w.sh`
    ``` shell
    sbatch align_to_t1w.sbatch
    ```

7. Finally, build the template:
    ``` shell
    sbatch hcph-template_build_template_mni_sl2.sbatch
    ```

8. Inspect the quality of the template by looking at the `A_tpl_template0.nii.gz` and `A_tpl_template1.nii.gz`.

9. *Optional:* Refine the template by running the script to build the template with the latest template as input (e.g. `-z my_initial_template.nii.gz`, see `antsMultivariateTemplateConstruction2.sh` documentation for examples)

### Running locally (this has not been tested but is here to provide tips on how to build your template locally)

Since all of the *SLURM* `.sbatch` scripts are just calling existing scripts, it could be easy to directly copy the calls from the `.sbatch` scripts to run in a local environment.