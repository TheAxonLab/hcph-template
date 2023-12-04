# HCPH Template: A high definition anatomical brain template of one individual healthy subject

## Installation

### Dependencies

To run the code you will need some external routines included in open-source packages:
- `N4BiasFieldCorrection`, `DenoiseImage` and `antsMultivariateTemplateConstruction2.sh` from the Advanced Normalization Tools [*ANTs*](https://github.com/ANTsX/ANTs).
- `mri_normalize` from [*FreeSurfer*](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_normalize).

## Running the code

This code has been developped to run on a high power computing (HPC) unit using the SLURM job manager.
Some tips are added to run the code locally.
Before running, make sure your anatomical data (T1w and T2w images) are stored in a BIDS dataset.

### Running in a *Slurm* environment (HPC)

#### Legacy step-by-step run

Start by adapting the `create_input_dir.sh` script to your system:
    
- You should modify the `bids_dir`, `sub`, `save_dir` and `template_name` variables to match your environments.

- If your anatomical images does not have the `desc-undistorted` tag, replaces the definition of `file_list` with a tag unique to your file names:
    
    ``` shell
    files_list=$( ls $sub_dir/ses-0*anat/*UNIQUE_TAG*.nii.gz )
    ```

Create the input directory:
``` shell
sh create_input_dir.sh
```

**IMPORTANT:** Check that the `templateInput.csv` file matches the T1w and T2w images in the output directory.

Create a `allImages.txt` file with all anatomical files to be processed:

```shell
ls /path/to/data/*.nii.gz >> allImages.txt
```

In your *SLURM* environment, run the denoising process:
``` shell
sbatch N4-Correction_Pipeline_cl2.sbatch
```

**IMPORTANT:** Check the quality of some images at each step.
It is crucial to validate the bias field correction (or correction for Intensity Non-Uniformities (INU)) as any reccurrent non-uniformity will be vsible on the template.
You may do it by looking at the `*corr.nii.gz` images and making sure there are no area that are particularly bright/dark.

Adjust the parameters of the bias field correction by update the following line in `N4-Correction-fortemplate.sh`:
``` shell
N4BiasFieldCorrection -d 3 -b [80] -i $imFile -o [$n4corrt1,$n4corrbf]
```

Finally, build the template:
``` shell
sbatch hcph-template_build_template_sl2.sbatch
```

Inspect the quality of the template by looking at the `A_tpl_template0.nii.gz` and `A_tpl_template1.nii.gz`.

*Optional:* Refine the template by running the script to build the template with the latest template as input (e.g. `-z my_initial_template.nii.gz`, see `antsMultivariateTemplateConstruction2.sh` documentation for examples)

### Running locally (this has not been tested but is here to provide tips on how to build your template locally)

Since all of the *SLURM* `.sbatch` scripts are just calling existing scripts, it could be easy to directly copy the calls from the `.sbatch` scripts to run in a local environment.