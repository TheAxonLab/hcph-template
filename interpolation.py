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
""" Python script to compute interpolation of 3D images onto a high-definition grid
based on the accuracy of images alignments (i.e. the distance between the projected
voxel center of the grid and the voxel center of the image).

Run as (see 'python compute_fc.py -h' for options):

    python interpolation.py path_to_data

"""

import numpy as np
import nibabel as nib
import os
import os.path as op

import logging

from joblib import Parallel, delayed

from typing import Union, Optional

from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_script

tqdm_func = tqdm_notebook

from scipy.io import loadmat
from scipy.interpolate import BSpline, _bsplines
from scipy.ndimage import map_coordinates

from nilearn.image import smooth_img
from skimage.filters import gaussian

from nilearn.datasets import load_mni152_template
from nitransforms.base import ImageGrid
from nitransforms import LinearTransformsMapping, Affine
from nitransforms.io.itk import ITKLinearTransform

from config import get_arguments, logger_config

# from nitransforms.nitransforms.linear import LinearTransformsMapping
# from nitransforms.nitransforms.base import ImageGrid


def get_anat_filenames(
    path_to_data: str,
    modality_filter: list = ["T1w"],
    pattern: str = "",
    template_prefix: str = "A_tpl",
    exclude: list = [],
) -> list:
    file_list = os.listdir(path_to_data)

    for fltrs in modality_filter + [pattern]:
        file_list = [file for file in file_list if fltrs in file]

    # Removing template files (e.g. with prefix "A_tpl")
    for tpl in [template_prefix] + exclude:
        file_list = [file for file in file_list if tpl not in file]

    anat_files_list = [op.join(path_to_data, file) for file in file_list]

    return sorted(anat_files_list)


def get_transform_files(
    path_to_data: str,
    transform_dir: str,
    transform_name: str = "Affine",
    template_prefix: str = "template",
) -> list[str]:
    path_to_transforms = op.join(path_to_data, transform_dir)

    transforms_files = [
        op.join(path_to_transforms, file)
        for file in os.listdir(path_to_transforms)
        if "Affine" in file and "template" not in file
    ]
    return sorted(transforms_files)


def generate_MNI_grid(
    resolution: float = 0.8, save_path: Optional[str] = None
) -> ImageGrid:
    """_summary_

    Parameters
    ----------
    resolution : float, optional
        _description_, by default 0.8

    Returns
    -------
    ImageGrid
        _description_
    """
    mni_template = load_mni152_template(resolution)

    if save_path is not None:
        print("Saving MNI Template")
        mni_filename = f"mni_template-res{resolution}mm.nii.gz"
        saveloc = op.join(save_path, mni_filename)

        if not op.isfile(saveloc):
            mni_template.to_filename(saveloc)

    mni_grid = ImageGrid(mni_template)
    return mni_grid


def mat2affine(files: Union[str, list, np.ndarray]) -> Union[list, np.ndarray]:
    """_summary_

    Parameters
    ----------
    files : Union[str, list, np.ndarray]
        _description_

    Returns
    -------
    Union[list, np.ndarray]
        _description_
    """
    if isinstance(files, (list, np.ndarray)):
        affines = [mat2affine(file) for file in files]
        return affines

    affine = ITKLinearTransform.from_filename(files)
    return affine.to_ras()


def get_transforms(
    transform_file_list: Union[str, np.ndarray],
    anat_file_list: Union[str, np.ndarray],
) -> list:
    affine_list = mat2affine(transform_file_list)

    transform_list = LinearTransformsMapping(affine_list)

    transforms_w_ref = []
    for trans, t_file, anat in zip(transform_list, transform_file_list, anat_file_list):
        # Safety check that anat files and transform files are corresponding
        if isinstance(anat, str) and op.basename(anat).split(".")[0] not in op.basename(
            t_file
        ):
            print(
                f"WARNING: {op.basename(anat).split('.')[0]} "
                f"not in {op.basename(t_file)}"
            )
        trans.reference = anat
        transforms_w_ref.append(trans)

    return transforms_w_ref


def ref_id_to_target_id(
    indices: Union[list, np.ndarray],
    ref: Union[ImageGrid, nib.Nifti1Image],
    target: Union[ImageGrid, nib.Nifti1Image],
    transform: LinearTransformsMapping,
    **kwargs,
) -> np.ndarray:
    coords_in_ref = ref.ras(indices)
    mapped_coords = transform.map(coords_in_ref, **kwargs)
    ras2vox = ~Affine(target.affine)

    return ras2vox.map(mapped_coords)


def dist_from_center(coords: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
    center_coords = np.floor(coords) + 0.5

    if center_coords.ndim > 1:
        dist = np.linalg.norm(coords - center_coords, axis=1)
    else:
        dist = np.linalg.norm(coords - center_coords)

    return dist


def get_voxel_center_dist(
    grid_ids: Union[list, np.ndarray],
    fixed_img: ImageGrid,
    moving_img: nib.Nifti1Image,
    transform: LinearTransformsMapping,
) -> np.ndarray:
    grid_center_ids = grid_ids + 0.5
    center_in_moving_ids = ref_id_to_target_id(
        grid_center_ids, fixed_img, moving_img, transform
    )

    distances = dist_from_center(center_in_moving_ids)

    return distances


def get_spline_kernel(order: int, limits: Optional[float] = None) -> _bsplines.BSpline:
    # MAX_DIST = 0.6
    MAX_DIST = np.sqrt(3 * 0.5**2)
    # MAX_DIST = 1

    if limits is None:
        limits = MAX_DIST
    knots = np.linspace(-limits, limits, order + 2)
    return BSpline.basis_element(knots)


def normalize_distances(
    distances: np.ndarray,
    dist_kernel_order: int = 1,
    normalize: bool = True,
    axis: int = 0,
    **kwargs,
) -> np.ndarray:
    spline_kernel = get_spline_kernel(dist_kernel_order, **kwargs)

    norm_dist = spline_kernel(distances)

    if np.any(norm_dist < 0):
        logging.warn("Negative values after BSpline kernel!")

    if normalize:
        return norm_dist / norm_dist.sum(axis=axis)
    return norm_dist


def sample_from_indices(
    indices: Union[list, np.ndarray],
    fixed_img: ImageGrid,
    moving_img: nib.Nifti1Image,
    transform: LinearTransformsMapping,
    **kwargs,
) -> np.ndarray:
    mapped_indices = ref_id_to_target_id(indices, fixed_img, moving_img, transform)

    # fwhm = 1.2
    # sigma = fwhm / np.sqrt(8 * np.log(2))
    ## nilearn_smoothed = smooth_img(moving_img, fwhm=fwhm)
    # gauss_smoothed = gaussian(moving_img.get_fdata(), sigma=sigma, mode="constant")

    # interpolated_values = map_coordinates(gauss_smoothed, mapped_indices.T, **kwargs)

    interpolated_values = map_coordinates(
        moving_img.get_fdata(), mapped_indices.T, **kwargs
    )
    return interpolated_values


def get_sample_val_and_dist(
    indices: Union[list, np.ndarray],
    fixed_img: ImageGrid,
    moving_list: list[str],
    transforms: list[LinearTransformsMapping],
    weight: bool = True,
    interpolate: bool = True,
    spline_order: int = 1,
) -> Union[tuple[np.ndarray], np.ndarray]:
    # Resampled average will be correct if `weight` is False
    distances = np.full((len(transforms), len(indices)), 1 / len(transforms))
    resampled = np.zeros_like(distances)
    for transform_id, (transform, moving) in enumerate(zip(transforms, moving_list)):
        moving_img = nib.load(moving)
        if weight:
            distances[transform_id] = get_voxel_center_dist(
                indices, fixed_img, moving_img, transform
            )

        if interpolate:
            resampled[transform_id] = sample_from_indices(
                indices, fixed_img, moving_img, transform, order=spline_order
            )

    if interpolate:
        return resampled, distances

    return distances


def interpolate_from_indices(
    indices: Union[list, np.ndarray],
    fixed_img: ImageGrid,
    moving_list: list[str],
    transforms: list[LinearTransformsMapping],
    weight: bool = True,
    interpolate: bool = True,
    spline_order: int = 1,
    **kwargs,
) -> np.ndarray:
    resampled, distances = get_sample_val_and_dist(
        indices, fixed_img, moving_list, transforms, weight, interpolate, spline_order
    )

    return resampled.__mul__(normalize_distances(distances, **kwargs)).sum(axis=0)


def batch_handler(
    input_array: np.ndarray,
    n_batches: int = 100,
    size: Optional[int] = None,
    limit: Optional[int] = None,
) -> list:
    if n_batches is not None and size is not None:
        print(
            "WARNING: both `size` and `n_batches` "
            "have been set - setting `size` to None."
        )
        size = None
    if size is not None:
        n_batches = np.ceil(len(input_array) / size).astype(int)
    if n_batches is not None:
        size = np.ceil(len(input_array) / n_batches).astype(int)
    if limit is None:
        limit = n_batches

    batch_indices = np.arange(n_batches)[:limit]

    return [input_array[i * size : (i + 1) * size] for i in batch_indices]


def distance_weighted_interpolation(
    fixed_img: ImageGrid,
    moving_list: list[str],
    transforms: list[LinearTransformsMapping],
    n_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    batch_limit: Optional[int] = None,
    weight: bool = True,
    normalize: bool = True,
    interpolate: bool = True,
    dist_kernel_order: int = 1,
    spline_order: int = 1,
    n_jobs: int = 1,
) -> np.ndarray:
    # Separate coordinates as n_batches of even size
    batches = batch_handler(fixed_img.ndindex.T, n_batches, batch_size, batch_limit)

    interpolated_values = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(interpolate_from_indices)(
            batch_indices,
            fixed_img,
            moving_list,
            transforms,
            weight=weight,
            normalize=normalize,
            interpolate=interpolate,
            spline_order=spline_order,
            dist_kernel_order=dist_kernel_order,
        )
        for batch_indices in batches
    )

    interpolated_array = np.zeros(fixed_img.shape)
    for values, indices in tqdm_func(
        zip(interpolated_values, batches), total=len(batches)
    ):
        interpolated_array[tuple(indices.T)] = values

    return interpolated_array
    # return np.hstack(interpolated_values).reshape(fixed_img.shape)


def get_distance_map(
    fixed_img: ImageGrid,
    moving_list: list[str],
    transforms: list[LinearTransformsMapping],
    n_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    batch_limit: Optional[int] = None,
    n_jobs: int = 1,
    **kwargs,
) -> np.ndarray:
    # Separate coordinates as n_batches of even size
    batches = batch_handler(fixed_img.ndindex.T, n_batches, batch_size, batch_limit)

    distance_map = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(get_sample_val_and_dist)(
            batch_indices,
            fixed_img,
            moving_list,
            transforms,
            weight=True,
            interpolate=False,
            **kwargs,
        )
        for batch_indices in batches
    )

    distances_array = np.zeros((len(transforms), *fixed_img.shape))
    for values, indices in tqdm_func(zip(distance_map, batches), total=len(batches)):
        distances_array[:, indices.T[0], indices.T[1], indices.T[2]] = values

    return distances_array


def get_individual_map(
    fixed_img: ImageGrid,
    moving_list: list[str],
    transforms: list[LinearTransformsMapping],
    n_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    batch_limit: Optional[int] = None,
    n_jobs: int = 1,
    **kwargs,
) -> np.ndarray:
    # Separate coordinates as n_batches of even size
    batches = batch_handler(fixed_img.ndindex.T, n_batches, batch_size, batch_limit)

    val_and_dist = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(get_sample_val_and_dist)(
            batch_indices,
            fixed_img,
            moving_list,
            transforms,
            weight=True,
            interpolate=True,
            **kwargs,
        )
        for batch_indices in batches
    )

    distances_array = np.zeros((len(transforms), *fixed_img.shape))
    images_array = np.zeros_like(distances_array)
    for values, indices in tqdm_func(zip(val_and_dist, batches), total=len(batches)):
        images_array[:, indices.T[0], indices.T[1], indices.T[2]] = values[0]
        distances_array[:, indices.T[0], indices.T[1], indices.T[2]] = values[1]

    return distances_array, images_array


def main():
    ####################
    # WORK IN PROGRESS #
    ####################

    args = get_arguments()

    path_to_data = args.data_dir
    save_dir = args.output
    transform_dir = args.transform_dir

    resolution = args.resolution
    weight = args.no_weight
    dist_kernel_order = args.d_order
    spline_order = args.bspline_order
    n_batches = args.n_batches
    n_jobs = args.n_jobs

    verbosity_level = args.verbosity
    logger_config(verbosity_level)

    logging.info(f"Generating MNI grid at resolution {resolution} mm")
    mni_grid = generate_MNI_grid(resolution)

    anat_files = get_anat_filenames(path_to_data)
    logging.info(f"{len(anat_files)} anat files found")

    transform_files = get_transform_files(path_to_data, transform_dir)
    transforms = get_transforms(transform_files, [mni_grid] * len(transform_files))

    logging.info(f"{len(transforms)} transforms found")

    interpolated_map = distance_weighted_interpolation(
        mni_grid,
        anat_files,
        transforms,
        n_batches=n_batches,
        weight=weight,
        normalize=True,
        interpolate=True,
        batch_limit=None,
        dist_kernel_order=dist_kernel_order,
        spline_order=spline_order,
        n_jobs=n_jobs,
    )

    if save_dir is not None:
        interpolated_image = nib.Nifti1Image(
            interpolated_map.astype(np.int16), affine=mni_grid.affine
        )
        # Save as int16 to reduce file size
        interpolated_image.header.set_data_dtype("int16")

        suffix = f"N{len(anat_files)}"
        suffix += weight * f"DisWei{dist_kernel_order}"
        suffix += "AllInRes_"

        # fname = f"DiWeTemplate_res-{resolution}_T1w.nii.gz"
        fname = f"distance_weighted_template_res-{resolution}_desc-{suffix}T1w.nii.gz"

        interpolated_image.to_filename(op.join(save_dir, fname))


if __name__ == "__main__":
    tqdm_func = tqdm_script
    main()
