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

Run as (see 'python interpolation.py -h' for options):

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

from scipy.interpolate import BSpline, _bsplines
from scipy.ndimage import map_coordinates

from nilearn.datasets import load_mni152_template
from nitransforms import LinearTransformsMapping, Affine
from nitransforms.base import ImageGrid

from nitransforms.manip import TransformChain
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
) -> list[str]:
    """Get the anatomical files including `pattern` and `modality_filter` but not
    including `template_prefix` and any str in `exclude`.

    Parameters
    ----------
    path_to_data : str
        path to the data directory
    modality_filter : list, optional
        modality to include, by default ["T1w"]
    pattern : str, optional
        pattern to match in the filename, by default ""
    template_prefix : str, optional
        pattern related to already computed templates to exclude in the filename, by
        default "A_tpl"
    exclude : list, optional
        pattern(s) to exclude in the filename, by default []

    Returns
    -------
    list[str]
        list of filenames (without directory !)
    """
    file_list = os.listdir(path_to_data)

    for fltrs in modality_filter + [pattern]:
        file_list = [file for file in file_list if fltrs in file]

    # Removing template files (e.g. with prefix "A_tpl")
    for tpl in [template_prefix] + exclude:
        file_list = [file for file in file_list if tpl not in file]

    anat_files_list = [op.join(path_to_data, file) for file in file_list]

    return sorted(anat_files_list)


def get_transform_files(
    transform_dir: str,
    transform_name: str = "Affine",
    template_pattern: str = "template",
    exclude: list = [],
) -> list[str]:
    """Get the transform files not including `template_pattern`.

    Parameters
    ----------
    transform_dir : str
        path to the directory with transforms
    transform_name : str, optional
        name of the transform, by default "Affine"
    template_prefix : str, optional
        template pattern to exclude in the file search, by default "template"
    exclude : list, optional
        pattern(s) to exclude in the filename, by default []

    Returns
    -------
    list[str]
        list of transform files
    """
    # path_to_transforms = op.join(path_to_data, transform_dir)

    transforms_files = [
        op.join(transform_dir, file)
        for file in os.listdir(transform_dir)
        if transform_name in file and template_pattern not in file
    ]

    # Removing files to be excluded
    for excl in exclude:
        transforms_files = [file for file in transforms_files if excl not in file]

    return sorted(transforms_files)


def generate_MNI_grid(
    resolution: float = 0.8, save_path: Optional[str] = None
) -> ImageGrid:
    """Create a 3D grid in MNI coordinates with a specific `resolution`.

    Parameters
    ----------
    resolution : float, optional
        grid resolution (in mm), by default 0.8
    save_path : Optional[str], optional
        path to save the grid, by default None

    Returns
    -------
    ImageGrid
        empty reference grid
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


def generate_grid_from_img(
    image: Union[str, ImageGrid, nib.Nifti1Image], resolution: float = 0.8
) -> ImageGrid:
    """Create a 3D grid with a specific `resolution` in the space of the specified
    `image`.

    Parameters
    ----------
    image : Union[str, ImageGrid, nib.Nifti1Image]
        input image to create the grid on
    resolution : float, optional
        grid resolution (in mm), by default 0.8

    Returns
    -------
    ImageGrid
        empty reference grid
    """

    if isinstance(image, str):
        image = nib.load(image)

    orig_res = image.header.get_zooms()[-1]
    scale_factor = orig_res / resolution

    if np.allclose(scale_factor, 1):
        return ImageGrid(image)

    new_shape = [int(scale_factor * s) for s in image.shape]

    empty_array = np.zeros((new_shape), dtype=np.uint8)
    new_affine = nib.affines.rescale_affine(
        image.affine, image.shape, zooms=resolution, new_shape=empty_array.shape
    )
    empty_grid = ImageGrid(nib.Nifti1Image(empty_array, new_affine))
    return empty_grid


def mat2affine(files: Union[str, list, np.ndarray]) -> Union[list, np.ndarray]:
    """Load `.mat` transforms files and returns the corresponding affine matrix.

    Parameters
    ----------
    files : Union[str, list, np.ndarray]
        path (or list of paths) to the transform files

    Returns
    -------
    Union[list, np.ndarray]
        affine matrix (or list of affine matrices)
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
    """Load the affine matrix of the transforms and assign the corresponding reference

    Parameters
    ----------
    transform_file_list : Union[str, np.ndarray]
        list of paths to the transform files
    anat_file_list : Union[str, np.ndarray]
        list of paths to the anatomical files

    Returns
    -------
    list
        list of LinearTransformsMapping with corresponding references
    """
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
    """Map the indices of the `ref` array onto the indices of the `target` array using
    the `transform`

    Parameters
    ----------
    indices : Union[list, np.ndarray]
        array indices to map
    ref : Union[ImageGrid, nib.Nifti1Image]
        reference image
    target : Union[ImageGrid, nib.Nifti1Image]
        target image to map coordinates to
    transform : LinearTransformsMapping
        transform to map the `ref` coordinates onto the `target`

    Returns
    -------
    np.ndarray
        mapped indices
    """
    coords_in_ref = ref.ras(indices)
    mapped_coords = transform.map(coords_in_ref, **kwargs)

    ras2vox = Affine(target.affine)
    return ras2vox.map(mapped_coords, inverse=True)


def dist_from_center(coords: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
    """Compute the distances between the coordinates in `coords` and the center of the
    nearest voxel.

    Since `coords` are supposed to be the indices of the reference voxel
    centers projected into the target space and converted back to indices (in the
    target array), the closest voxel will have indices `np.floor(coords)`. For example,
    coordinates projected at indices (10.5, 15.2, 8.7) will be inside the voxel indexed
    at (10, 15, 8) with center (10.5, 15.5, 8.5). In this case, the distance is the
    norm of (0, 0.3, 0.2).

    Parameters
    ----------
    coords : Union[list, np.ndarray]
        input coordinates

    Returns
    -------
    Union[list, np.ndarray]
        computed distances
    """
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
    """Get the distance between the center of the voxels indexed by `grid_ids`
    projected as indices in `moving_img` and the center of the nearest voxel in
    `moving_img`.

    Parameters
    ----------
    grid_ids : Union[list, np.ndarray]
        indices of the reference voxels
    fixed_img : ImageGrid
        reference image grid
    moving_img : nib.Nifti1Image
        target image
    transform : LinearTransformsMapping
        transform from the reference to the target

    Returns
    -------
    np.ndarray
        array of computed distances
    """
    grid_center_ids = grid_ids + 0.5
    center_in_moving_ids = ref_id_to_target_id(
        grid_center_ids, fixed_img, moving_img, transform
    )

    distances = dist_from_center(center_in_moving_ids)

    return distances


def get_spline_kernel(order: int, limits: Optional[float] = None) -> _bsplines.BSpline:
    """Compute a BSpline basis function of order `order` in the range
    (-`limits`, `limits`).

    Parameters
    ----------
    order : int
        order of the BSpline basis
    limits : Optional[float], optional
        range of definition of the BSpline function such that
        BSpline(limits) = BSpline(-limits) = 0, by default None

    Returns
    -------
    _bsplines.BSpline
        BSpline kernel function
    """
    # MAX_DIST = 0.6
    MAX_DIST = np.sqrt(3 * 0.5**2)
    # MAX_DIST = 1

    if limits is None:
        limits = MAX_DIST

    if order == 1:

        def linear_kernel(x, zero=MAX_DIST):
            return -x / zero + 1

        return linear_kernel

    knots = np.linspace(-limits, limits, order + 2)
    return BSpline.basis_element(knots, extrapolate=False)


def normalize_distances(
    distances: np.ndarray,
    dist_kernel_order: int = 1,
    normalize: bool = True,
    axis: int = 0,
    offset: float = 0,
    **kwargs,
) -> np.ndarray:
    """Apply a BSpline basis kernel to the distances and normalizes them such that the
    sum of distances for one singel voxel is one.

    Parameters
    ----------
    distances : np.ndarray
        array of distances
    dist_kernel_order : int, optional
        order of the BSpline basis kernel, by default 1
    normalize : bool, optional
        condition to normalize the distances, by default True
    axis : int, optional
        axis to normalize (should be the axis of the number of distances for each voxel)
        , by default 0

    Returns
    -------
    np.ndarray
        normalized distances
    """
    spline_kernel = get_spline_kernel(dist_kernel_order, **kwargs)

    norm_dist = spline_kernel(distances)
    norm_dist = np.clip(spline_kernel(distances) + offset, a_max=1, a_min=0)

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
    """Sample values in the target (`moving_img`) for the `indices` of the reference
    (`fixed_img`) projected using the transform (`transform`).

    Parameters
    ----------
    indices : Union[list, np.ndarray]
        indices of the reference to sample on the target
    fixed_img : ImageGrid
        reference image
    moving_img : nib.Nifti1Image
        target image
    transform : LinearTransformsMapping
        transform between the reference coordinates to the target coordinates

    Returns
    -------
    np.ndarray
        array of sampled values
    """
    mapped_indices = ref_id_to_target_id(indices, fixed_img, moving_img, transform)

    # fwhm = 1.2
    # sigma = fwhm / np.sqrt(8 * np.log(2))
    # nilearn_smoothed = smooth_img(moving_img, fwhm=fwhm)
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
    spline_order: int = 3,
) -> Union[tuple[np.ndarray], np.ndarray]:
    """Sample the values and the distances to the closest voxel in all of the target
    images in `moving_list` for the input `indices` of the reference image
    (`fixed_img`).

    Parameters
    ----------
    indices : Union[list, np.ndarray]
        indices in the reference image
    fixed_img : ImageGrid
        reference image
    moving_list : list[str]
        list of target images
    transforms : list[LinearTransformsMapping]
        list of transforms from the reference to the targets
    weight : bool, optional
        condition to compute and normalize the distances, by default True
    interpolate : bool, optional
        condition to sample the values using BSpline interpolation, by default True
    spline_order : int, optional
        order of the BSpline interpolation, by default 3

    Returns
    -------
    Union[tuple[np.ndarray], np.ndarray]
        either a tuple of the resampled values with the corresponding distances or only
        a vector with the distances
    """
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
    spline_order: int = 3,
    **kwargs,
) -> np.ndarray:
    """Compute the distance weighted interpolation for the list of indices in input.

    Parameters
    ----------
    indices : Union[list, np.ndarray]
        indices to interpolate
    fixed_img : ImageGrid
        reference image
    moving_list : list[str]
        list of moving images
    transforms : list[LinearTransformsMapping]
        list of transforms from the reference to the targets
    weight : bool, optional
        condition to compute and normalize the distances, by default True
    interpolate : bool, optional
        condition to sample the values using BSpline interpolation, by default True
    spline_order : int, optional
        order of the BSpline interpolation, by default 3

    Returns
    -------
    np.ndarray
        resampled array weighted by the distance to the voxel center closest to the
        projected indices
    """
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
    """Separate the coordinates of `input_array` as batches (`n_batches` batches of
    size `size`) of coordinates (to parallelize computation and reduce memory usage)

    Parameters
    ----------
    input_array : np.ndarray
        array to compute coordinates from
    n_batches : int, optional
        number of batches, by default 100
    size : Optional[int], optional
        size of the batches - setting both `n_batches` and `size` will raise a warning,
        by default None
    limit : Optional[int], optional
        limit the number of batches to compute (for debug), by default None

    Returns
    -------
    list
        list of batches
    """
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
    offset: float = 0,
    spline_order: int = 1,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute the distance weighted interpolation of the target images in `moving_list`
    onto the reference image (`fixed_img`).

    Parameters
    ----------
    fixed_img : ImageGrid
        reference image
    moving_list : list[str]
        list of moving images
    transforms : list[LinearTransformsMapping]
        list of transforms from the reference to the targets
    n_batches : Optional[int], optional
        number of batches to send in parallel, by default None
    batch_size : Optional[int], optional
        size of the batches to send in parallel - setting both `n_batches` and `size`
        will raise a warning, by default None
    batch_limit : Optional[int], optional
        limit the number of batches to compute (for debug), by default None
    weight : bool, optional
        condition to compute the distances, by default True
    normalize : bool, optional
        condition to normalize the distances, by default True
    interpolate : bool, optional
        condition to sample the values using BSpline interpolation, by default True
    dist_kernel_order : int, optional
        order of the BSpline basis kernel to apply to the distances, by default 1
    spline_order : int, optional
        order of the BSpline interpolation, by default 3
    n_jobs : int, optional
        number of `Joblib` jobs to send in parallel, by default 1

    Returns
    -------
    np.ndarray
        interpolated image as an array
    """
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
            offset=offset,
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
    """Only computes the distance maps.

    Parameters
    ----------
    fixed_img : ImageGrid
        reference image
    moving_list : list[str]
        list of moving images
    transforms : list[LinearTransformsMapping]
        list of transforms from the reference to the targets
    n_batches : Optional[int], optional
        number of batches to send in parallel, by default None
    batch_size : Optional[int], optional
        size of the batches to send in parallel - setting both `n_batches` and `size`
        will raise a warning, by default None
    batch_limit : Optional[int], optional
        limit the number of batches to compute (for debug), by default None
    n_jobs : int, optional
        number of `Joblib` jobs to send in parallel, by default 1

    Returns
    -------
    np.ndarray
        array of distance maps
    """
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
    map_dtype: type = int,
    return_normalized: int = 0,
    n_jobs: int = 1,
    **kwargs,
) -> tuple[np.ndarray]:
    """Compute and return the resampled and distance map without applying the distance
    weighted interpolation.

    Parameters
    ----------
    fixed_img : ImageGrid
        reference image
    moving_list : list[str]
        list of moving images
    transforms : list[LinearTransformsMapping]
        list of transforms from the reference to the targets
    n_batches : Optional[int], optional
        number of batches to send in parallel, by default None
    batch_size : Optional[int], optional
        size of the batches to send in parallel - setting both `n_batches` and `size`
        will raise a warning, by default None
    batch_limit : Optional[int], optional
        limit the number of batches to compute (for debug), by default None
    map_dtype : type
        type of the interpolated map, default int
    return_normalized : int
        order of the BSpline for the distance normalization (0 means no normalization),
        default 0
    n_jobs : int, optional
        number of `Joblib` jobs to send in parallel, by default 1

    Returns
    -------
    tuple[np.ndarray]
        array of the distance maps and array of the resampled images
    """
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
    images_array = np.zeros_like(distances_array, dtype=map_dtype)
    for values, indices in tqdm_func(zip(val_and_dist, batches), total=len(batches)):
        images_array[:, indices.T[0], indices.T[1], indices.T[2]] = values[0]
        if return_normalized:
            distances_array[:, indices.T[0], indices.T[1], indices.T[2]] = (
                normalize_distances(values[1], dist_kernel_order=return_normalized)
            )

    return distances_array, images_array


def save_array_as_image(
    array: np.ndarray,
    affine: np.ndarray,
    save_dir: str,
    fname_pattern: str,
    fname_fills: dict,
    rescale: int = 255,
    img_dtype: type = np.int16,
) -> None:

    # Rescale the array between 0 and the rescale value
    if rescale > 0:
        array = (array / array.max() * rescale).astype(img_dtype)
        array[array < 0] = 0

    interpolated_image = nib.Nifti1Image(array, affine=affine)

    # Save as int16 to reduce file size
    dtype_str = str(img_dtype).split("'")[1].split(".")[-1]
    interpolated_image.header.set_data_dtype(dtype_str)

    fname = fname_pattern.format(**fname_fills)

    os.makedirs(save_dir, exist_ok=True)
    interpolated_image.to_filename(op.join(save_dir, fname))


def main():
    ####################
    # WORK IN PROGRESS #
    ####################

    args = get_arguments()

    path_to_data = args.data_dir
    path_to_template = args.template_dir
    save_dir = args.output
    image_modality = args.modality
    transform_dir = args.transform_dir
    exclude_ses = args.exclude_ses

    pre_transform_pattern = args.pre_transform
    post_transform_pattern = args.post_transform

    resolution = args.resolution
    weight = args.no_weight
    dist_kernel_order = args.d_order
    offset = args.offset
    spline_order = args.bspline_order
    n_batches = args.n_batches
    n_jobs = args.n_jobs
    use_mni = args.use_mni
    get_map = args.maps
    n_subset = args.n_subset

    verbosity_level = args.verbosity
    logger_config(verbosity_level)

    anat_files = get_anat_filenames(
        path_to_data,
        modality_filter=[image_modality],
        pattern=".nii.gz",
        exclude=exclude_ses,
    )
    anat_files = anat_files[:n_subset]
    logging.info(f"{len(anat_files)} anat files found")

    if use_mni:
        logging.info(f"Generating MNI grid at resolution {resolution} mm")
        mni_grid = generate_MNI_grid(resolution)
    elif image_modality != "T1w":
        logging.info(
            f"Generating grid from the 1st T1w image at resolution {resolution} mm"
        )
        mni_grid = generate_grid_from_img(
            anat_files[0].replace(image_modality, "T1w"), resolution
        )
    else:
        logging.info(
            f"Generating grid from the 1st image at resolution {resolution} mm"
        )
        mni_grid = generate_grid_from_img(anat_files[0], resolution)

    transform_files = get_transform_files(
        op.join(path_to_template, transform_dir), exclude=exclude_ses
    )
    transforms = get_transforms(transform_files, [mni_grid] * len(transform_files))
    transforms = transforms[:n_subset]

    logging.info(f"{len(transforms)} transforms found")

    pre_transforms = [Affine() for _ in range(len(transforms))]
    if pre_transform_pattern is not None:
        pre_transforms_files = get_transform_files(
            path_to_template,
            transform_name=pre_transform_pattern,
            exclude=exclude_ses,
        )
        ref_anat_files = get_anat_filenames(
            path_to_data,
            modality_filter=["T1w"],
            pattern=".nii.gz",
            exclude=exclude_ses,
        )
        pre_transforms = get_transforms(
            pre_transforms_files,
            [generate_grid_from_img(anat) for anat in ref_anat_files],
        )
        pre_transforms = pre_transforms[:n_subset]
        logging.info(f"{len(pre_transforms)} PRE transforms found")
    elif image_modality != "T1w":
        logging.warning(f"`{image_modality}` has been selected without pre-transforms!")

    post_transform = [Affine()]
    if post_transform_pattern is not None:
        if use_mni:
            post_transform_files = get_transform_files(
                path_to_template,
                transform_name=post_transform_pattern,
                template_pattern="^^^^",
                exclude=exclude_ses,
            )
            post_transform = get_transforms(
                post_transform_files, [mni_grid] * len(post_transform_files)
            )
            logging.info(f"{len(post_transform)} POST transforms found")
        else:
            logging.warning("Post-transforms should only be used with MNI grid!")
    else:
        post_transform_pattern = ""

    post_transform_list = post_transform * len(transforms)

    all_transforms = [
        TransformChain(transforms=[post, trans, pre])
        for pre, trans, post in zip(pre_transforms, transforms, post_transform_list)
    ]

    if get_map:
        logging.info("Computing individual maps ...")
        out_maps = get_individual_map(
            mni_grid,
            anat_files,
            all_transforms,
            n_batches=n_batches,
            batch_limit=None,
            spline_order=spline_order,
            return_normalized=True,
            n_jobs=n_jobs,
        )
    else:
        logging.info("Running distance weighted interpolation ...")
        out_maps = distance_weighted_interpolation(
            mni_grid,
            anat_files,
            all_transforms,
            n_batches=n_batches,
            weight=weight,
            normalize=True,
            interpolate=True,
            batch_limit=None,
            dist_kernel_order=dist_kernel_order,
            offset=offset,
            spline_order=spline_order,
            n_jobs=n_jobs,
        )

    if save_dir is not None:
        if isinstance(out_maps, tuple):
            for maps, map_name, map_type in zip(
                out_maps,
                ["dmapNorm", "interp"],
                [np.float32, np.int16],
            ):
                for image, img_name in zip(maps, anat_files):
                    fname_pattern = (
                        "sub-{sub}_ses-{ses}_res-{res}_desc-{desc}_{mod}.{ext}"
                    )
                    ses = op.basename(img_name).split("_")[1].split("-")[1]
                    sub = op.basename(img_name).split("_")[0].split("-")[1]
                    fname_fills = {
                        "sub": sub,
                        "ses": ses,
                        "res": resolution,
                        "desc": map_name,
                        "mod": image_modality,
                        "ext": "nii.gz",
                    }
                    save_array_as_image(
                        image,
                        affine=mni_grid.affine,
                        save_dir=save_dir,
                        fname_pattern=fname_pattern,
                        fname_fills=fname_fills,
                        img_dtype=map_type,
                    )
        else:
            space_name = "native"
            if use_mni:
                space_name = post_transform_pattern.split("_")[-1]
            suffix = (
                f"N{len(anat_files)}"
                + weight * f"DisWei{dist_kernel_order}"
                + (offset != 0) * f"Off{offset}"
            )
            fname_pattern = (
                "distance_weighted_template_res-{res}_space-{space}"
                + "_desc-{desc}{modality}.nii.gz"
            )
            fname_fills = {
                "res": resolution,
                "desc": suffix,
                "modality": image_modality,
                "space": space_name,
            }
            save_array_as_image(
                out_maps,
                affine=mni_grid.affine,
                save_dir=save_dir,
                fname_pattern=fname_pattern,
                fname_fills=fname_fills,
                img_dtype=np.int16,
            )


if __name__ == "__main__":
    tqdm_func = tqdm_script
    main()
