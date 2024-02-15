# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The Axon Lab <theaxonlab@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
""" Python script to normalize the intensity of 3D images by matching the image
histogram with a reference (usually the first image).

Run as (see 'python histomatch.py -h' for options):

    python histomatch.py path_to_data path_to_output

"""

import os
import os.path as op
import argparse
import logging
from typing import Optional, Union

import numpy as np
import nibabel as nib
from skimage.exposure import match_histograms
from joblib import Parallel, delayed

from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_script

import interpolation as interp


def get_arguments() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="""Normalize the intensity of 3D images by matching the image
        histogram with a reference (usually the first image).""",
    )

    # Input/Output arguments and options
    parser.add_argument("data_dir", help="path to data directory")

    parser.add_argument(
        "-o", "--output", default=None, type=str, help="path to save data"
    )

    parser.add_argument(
        "-m",
        "--modality",
        default=["T1w", "T2w"],
        action="store",
        nargs="+",
        help="a space delimited list of modularity(es)",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        default="N4corrden",
        type=str,
        help="a string pattern to identify files",
    )

    parser.add_argument(
        "-n",
        "--n_jobs",
        default=1,
        type=int,
        help="number of job(s) to send to `joblib`",
    )

    args = parser.parse_args()
    return args


def histomatch_single_img(
    image: Union[str, nib.Nifti1Image], img_ref: np.ndarray
) -> nib.Nifti1Image:
    """Normalize intensity of `image` to the one of `img_ref` using histogram matching.

    Parameters
    ----------
    image : Union[str, nib.Nifti1Image]
        image to normalize
    img_ref : np.ndarray
        reference image

    Returns
    -------
    nib.Nifti1Image
        normalized image
    """
    if isinstance(image, str):
        image = nib.load(image)
    matched_data = match_histograms(image.get_fdata(), img_ref)

    return nib.Nifti1Image(matched_data, affine=image.affine, dtype="uint8")


def histomatch_list(
    img_ref: np.ndarray,
    img_list: list[str],
    pattern: str = "N4corrden",
    saveloc: Optional[str] = None,
    n_jobs: int = 1,
) -> None:
    """Normalize a list of images to a reference image using histogram matching.

    Parameters
    ----------
    img_ref : np.ndarray
        reference image
    img_list : list[str]
        list of path to the images
    pattern : str, optional
        pattern to identify the image (used to rename the output), by default
        "N4corrden"
    saveloc : Optional[str], optional
        path to the output directory, by default None
    n_jobs : int, optional
        number of jobs to send to `joblib`, by default 1
    """

    if n_jobs > 1:
        logging.info(f"Running in parallel with {n_jobs} jobs")
    matched_images = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(histomatch_single_img)(file, img_ref) for file in img_list
    )

    for matched_img, file in tqdm_func(
        zip(matched_images, img_list), total=len(img_list)
    ):
        savename = op.basename(file).replace(pattern, pattern + "hist")
        matched_img.to_filename(op.join(saveloc, savename))


def main():

    args = get_arguments()

    path_to_imgs = args.data_dir
    saveloc = args.output

    modalities = args.modality
    pattern = args.pattern

    n_jobs = args.n_jobs

    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=1,
    )

    logging.captureWarnings(True)

    if saveloc is None:
        saveloc = path_to_imgs + "-histomatch"

    os.makedirs(saveloc, exist_ok=True)

    for modality in modalities:
        logging.info(f"Matching histograms for modality: {modality}")
        files_list = interp.get_anat_filenames(
            path_to_imgs,
            pattern=pattern,
            modality_filter=[modality],
            exclude=["in0048", "rerun"],
        )

        first_img = nib.load(files_list[0])
        first_data = first_img.get_fdata()

        histomatch_list(
            first_data, files_list, pattern=pattern, saveloc=saveloc, n_jobs=n_jobs
        )


if __name__ == "__main__":
    tqdm_func = tqdm_script
    main()
