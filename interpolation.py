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
""" Python script to denoise and aggregate timeseries and, using the latter, compute
functional connectivity matrices from BIDS derivatives (e.g. fmriprep).

Run as (see 'python compute_fc.py -h' for options):

    python compute_fc.py path_to_BIDS_derivatives

In the context of HCPh (pilot), it would be:

    python compute_fc.py /data/datasets/hcph-pilot/derivatives/fmriprep-23.1.4/
"""

import numpy as np
import pandas as pd
import nibabel as nib
import os

from itertools import product
from typing import Union, Optional

from scipy.io import loadmat

from nilearn.datasets import load_mni152_template
from nitransforms.base import ImageGrid
from nitransforms import LinearTransformsMapping

# from nitransforms.nitransforms.linear import LinearTransformsMapping
# from nitransforms.nitransforms.base import ImageGrid


def generate_MNI_grid(resolution: float = 0.8) -> ImageGrid:
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
    mni_grid = ImageGrid(mni_template)
    return mni_grid


def mat2affine(
    files: Union[str, list, np.ndarray], return_transform: bool = False
) -> Union[list, np.ndarray]:
    """_summary_

    Parameters
    ----------
    files : Union[str, list, np.ndarray]
        _description_
    return_transform : bool, optional
        _description_, by default False

    Returns
    -------
    Union[list, np.ndarray]
        _description_
    """
    AFFINE_LAST_ROW = [0, 0, 0, 1]
    ROTZOOM_SHAPE = (3, 3)

    if type(files) in [list, np.ndarray]:
        affines = [mat2affine(file) for file in files]
        if return_transform:
            return LinearTransformsMapping(affines)
        return affines

    mat_dict = loadmat(files)

    first_key = list(mat_dict.keys())[0]
    rot_zooms = mat_dict[first_key][:-3].reshape(ROTZOOM_SHAPE)
    translation = mat_dict[first_key][-3:]

    affine = np.vstack([np.hstack([rot_zooms, translation]), AFFINE_LAST_ROW])
    if return_transform:
        return LinearTransformsMapping([affine])
    return affine


def map_coordinates(
    coords: Union[list, np.ndarray],
    anat_img: Union[list, nib.Nifti1Image],
    grid_img: nib.Nifti1Image,
) -> Union[list, np.ndarray]:
    """Map the input coordinates from the space given by `anat_img` onto the space
    given by `grid_img`.

    Parameters
    ----------
    coords : Union[list, np.ndarray]
        _description_
    anat_img : Union[list, nib.Nifti1Image]
        _description_
    grid_img : nib.Nifti1Image
        _description_

    Returns
    -------
    list
        _description_
    """

    grid_vox2anat_vox = np.linalg.inv(anat_img.affine).dot(grid_img.affine)
    coords_target = nib.affines.apply_affine(grid_vox2anat_vox, coords)

    return coords_target


def get_anat_filenames(
    path_to_data: list[str], modality_filter: list = ["T1w"], pattern: str = ""
) -> list:
    file_list = os.listdir(path_to_data)

    for fltrs in modality_filter + [pattern]:
        file_list = [file for file in file_list if fltrs in file]

    # Removing template files (e.g. with prefix "A_tpl")
    template_prefix = "A_tpl"
    anat_files_list = [file for file in file_list if template_prefix not in file]

    return sorted(anat_files_list)


def get_boundary_in_target(
    img: nib.Nifti1Image,
    target: nib.Nifti1Image,
    origin: Union[list, np.ndarray] = [0, 0, 0],
    v: bool = False,
) -> np.ndarray:
    img_boundaries = np.array([origin, img.shape])
    img_boundaries[-1] -= 1

    img_extremes = list(
        product(img_boundaries[:, 0], img_boundaries[:, 1], img_boundaries[:, 2])
    )

    extremes_inTarget = np.array(map_coordinates(img_extremes, target, img))
    if v:
        print(f"Computed extremes are: \n{extremes_inTarget}")
    extremes_inTarget[extremes_inTarget < 0] = 0
    extremes_inTarget = np.ceil(extremes_inTarget) + 1
    for i in range(3):
        extremes_inTarget[:, i] = extremes_inTarget[:, i].clip(0, target.shape[i])

    boundaries_inTarget = np.vstack(
        [extremes_inTarget.min(axis=0), extremes_inTarget.max(axis=0)]
    )

    return boundaries_inTarget


def consensus_boundary(file_list: list, target: nib.Nifti1Image) -> np.ndarray:
    all_boundaries = np.zeros((len(file_list), 2, 3))
    for file_id, file in enumerate(file_list):
        img = nib.load(file)

        all_boundaries[file_id] = get_boundary_in_target(img, target)

    boundary = np.vstack(
        [all_boundaries.min(axis=(0, 1)), all_boundaries.max(axis=(0, 1))]
    ).astype(int)
    print(f"Consensus boundary is: \n {boundary}")
    return boundary


def distance_from_closest_center(
    coords: Union[list, np.ndarray], return_array_id: bool = True
) -> Union[float, tuple[float, np.ndarray]]:
    array_id = np.floor(coords).astype(int)

    # Vector from voxel center to coordinates
    diff = coords - array_id - 0.5

    if array_id.ndim > 1:
        dist = np.linalg.norm(diff, axis=1)
    else:
        dist = np.linalg.norm(diff)

    if return_array_id:
        return dist, array_id

    return dist


def dist_weighted_sampling(
    img: nib.Nifti1Image,
    target: nib.Nifti1Image,
    coords: Union[str, np.ndarray],
    weight: float = 0.3,
    # return_dist: bool = False,
) -> list[np.ndarray, np.ndarray]:
    # ) -> Union[np.ndarray, list[np.ndarray]]:
    img_array = img.get_fdata()
    grid_in_anat = map_coordinates(coords, img, target)

    sampled_array = np.zeros(len(coords))
    dist_array = np.zeros_like(sampled_array)
    # sampled_array_weighted = np.zeros_like(sampled_array)

    # Checking that all coordinates are within the img array by building a mask
    in_boundary_mask = np.all(grid_in_anat < np.array(img_array.shape), axis=1)

    distances, array_ids = distance_from_closest_center(grid_in_anat[in_boundary_mask])
    array_to_index = tuple([array_ids.T[i] for i in range(3)])

    # Largest distance from the center of the pixel is 0.5 in each dimension
    # sqrt(0.5² + 0.5² + 0.5²)
    # max_dist = np.sqrt(3 * (0.5**2))
    # norm_dist = (1 - weight) + weight * (1 - distances / max_dist)

    sampled_array[in_boundary_mask] = img_array[array_to_index]
    dist_array[in_boundary_mask] = distances
    # sampled_array_weighted[in_boundary_mask] = img_array[array_to_index] * norm_dist

    # if return_unweighted:
    #    return sampled_array_weighted, sampled_array

    return sampled_array, dist_array


def dist_weighted_interpolation(
    file_list: list,
    target: nib.Nifti1Image,
    slab_size: int = 10,
    boundary: Optional[np.ndarray] = None,
    z_coords_range: Optional[list] = None,
    v: bool = False,
    **kwargs,
) -> np.ndarray:
    VOXEL_CENTER = 0.5

    if boundary is None:
        boundary = consensus_boundary(file_list, target)

    dim = [np.arange(bound_limits[0], bound_limits[1]) for bound_limits in boundary.T]

    n_slabs = np.ceil(len(dim[2]) / slab_size).astype(int)
    slab_split = np.array_split(dim[2], n_slabs)

    if v:
        print(
            f"Boundary has size {len(dim[0])}x{len(dim[1])}x{len(dim[2])} "
            f"which is equal to {len(dim[0])*len(dim[1])*len(dim[2])} voxels."
        )
        slab_lengths = [len(slab) for slab in slab_split]
        slab_df = pd.Series(slab_lengths).value_counts()
        print("Data has been split in:")
        print(
            ", ".join(
                [f"{n_s} slabs of size {s_size}" for s_size, n_s in slab_df.items()]
            )
        )

    interp_array = np.zeros(target.shape)
    interp_unweighted = np.zeros_like(interp_array)

    for z_coords in slab_split[:3]:
        slab_coords = [
            (prod[0], prod[1], prod[2]) for prod in product(dim[0], dim[1], z_coords)
        ]
        slab_coords = np.array(slab_coords)
        slab_coords_ids = (slab_coords.T[0], slab_coords.T[1], slab_coords.T[2])
        slab_mid_coords = np.array(slab_coords) + VOXEL_CENTER
        if v:
            print(f"Slice has {len(slab_coords)} voxels.")

        slab_values = np.zeros(
            (len(file_list), len(dim[0]), len(dim[1]), len(z_coords))
        )
        slab_dist = np.zeros_like(slab_values)

        print(slab_values.shape)
        print(slab_coords_ids[0])

        for file_id, file in enumerate(file_list):
            img = nib.load(file)
            (
                slab_values[file_id][slab_coords_ids],
                slab_dist[file_id][slab_coords_ids],
            ) = dist_weighted_sampling(img, target, slab_mid_coords)

    interpolated_img = None
    return interpolated_img
