# T1 Maps generation module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


import json
import logging
import numpy as np
import scipy
import scipy.interpolate
import skimage
import tqdm
import nibabel
from functools import partial
from typing import Optional
from pathlib import Path

from ..data.base import MRIData
from ..data.io import load_mri_data, save_mri_data
from ..masking.masks import create_csf_mask
from .utils import (
    mri_facemask,
    fit_voxel,
    nan_filter_gaussian,
    T1_lookup_table,
)

logger = logging.getLogger(__name__)


def compute_looklocker_t1_array(data: np.ndarray, time_s: np.ndarray, t1_roof: float = 10000.0) -> np.ndarray:
    """
    Computes T1 relaxation maps from Look-Locker data using Levenberg-Marquardt fitting.

    Args:
        data (np.ndarray): 4D numpy array (x, y, z, time) of Look-Locker MRI signals.
        time_s (np.ndarray): 1D array of trigger times in seconds.
        t1_roof (float, optional): Maximum allowed T1 value (ms) to cap spurious fits. Defaults to 10000.0.

    Returns:
        np.ndarray: 3D numpy array representing the T1 map in milliseconds. Voxels
        that fail to fit or fall outside the mask are set to NaN.
    """
    assert len(data.shape) >= 4, f"Data should be at least 4-dimensional, got shape {data.shape}"
    mask = mri_facemask(data[..., 0])
    valid_voxels = (np.nanmax(data, axis=-1) > 0) & mask

    data_normalized = np.nan * np.zeros_like(data)
    # Prevent divide by zero warnings dynamically
    max_vals = np.nanmax(data, axis=-1)[valid_voxels, np.newaxis]
    data_normalized[valid_voxels] = data[valid_voxels] / max_vals

    voxel_mask = np.array(np.where(valid_voxels)).T
    d_masked = np.array([data_normalized[i, j, k] for (i, j, k) in voxel_mask])

    with tqdm.tqdm(total=len(d_masked), desc="Fitting Look-Locker Voxels") as pbar:
        voxel_fitter = partial(fit_voxel, time_s, pbar)
        vfunc = np.vectorize(voxel_fitter, signature="(n) -> (3)")
        fitted_coefficients = vfunc(d_masked)

    x2 = fitted_coefficients[:, 1]
    x3 = fitted_coefficients[:, 2]

    i, j, k = voxel_mask.T
    t1map = np.nan * np.zeros_like(data[..., 0])

    # Calculate T1 in ms. Formula: T1 = (x2 / x3)^2 * 1000
    t1map[i, j, k] = (x2 / x3) ** 2 * 1000.0

    return np.minimum(t1map, t1_roof)


def create_largest_island_mask(data: np.ndarray, radius: int = 10, erode_dilate_factor: float = 1.3) -> np.ndarray:
    """
    Creates a binary mask isolating the largest contiguous non-NaN region in an array.

    Args:
        data (np.ndarray): The 3D input data containing NaNs and valid values.
        radius (int, optional): The radius for morphological dilation. Defaults to 10.
        erode_dilate_factor (float, optional): Multiplier for the erosion radius
            relative to the dilation radius. Defaults to 1.3.

    Returns:
        np.ndarray: A boolean 3D mask of the largest contiguous island.
    """
    mask = skimage.measure.label(np.isfinite(data))
    regions = skimage.measure.regionprops(mask)
    if not regions:
        return np.zeros_like(data, dtype=bool)

    regions.sort(key=lambda x: x.num_pixels, reverse=True)
    mask = mask == regions[0].label
    try:
        skimage.morphology.remove_small_holes(mask, max_size=10 ** (mask.ndim), connectivity=2, out=mask)
    except TypeError:
        # Older versions of skimage use area_threshold instead of max_size
        skimage.morphology.remove_small_holes(mask, area_threshold=10 ** (mask.ndim), connectivity=2, out=mask)
    skimage.morphology.dilation(mask, skimage.morphology.ball(radius), out=mask)
    skimage.morphology.erosion(mask, skimage.morphology.ball(erode_dilate_factor * radius), out=mask)
    return mask


def remove_outliers(data: np.ndarray, mask: np.ndarray, t1_low: float, t1_high: float) -> np.ndarray:
    """
    Applies a mask and removes values outside the physiological T1 range.

    Args:
        data (np.ndarray): 3D array of T1 values.
        mask (np.ndarray): 3D boolean mask of the brain/valid area.
        t1_low (float): Lower physiological limit.
        t1_high (float): Upper physiological limit.

    Returns:
        np.ndarray: A cleaned 3D array with outliers and unmasked regions set to NaN.
    """
    processed = data.copy()
    processed[~mask] = np.nan
    outliers = (processed < t1_low) | (processed > t1_high)
    processed[outliers] = np.nan
    return processed


def looklocker_t1map_postprocessing(
    T1map: Path,
    T1_low: float,
    T1_high: float,
    radius: int = 10,
    erode_dilate_factor: float = 1.3,
    mask: Optional[np.ndarray] = None,
    output: Path | None = None,
) -> MRIData:
    """I/O wrapper for masking, outlier removal, and NaN filling on a T1 map."""
    t1map_mri = load_mri_data(T1map, dtype=np.single)
    t1map_data = t1map_mri.data.copy()

    if mask is None:
        mask = create_largest_island_mask(t1map_data, radius, erode_dilate_factor)

    t1map_data = remove_outliers(t1map_data, mask, T1_low, T1_high)

    if np.isfinite(t1map_data).sum() / t1map_data.size < 0.01:
        raise RuntimeError("After outlier removal, less than 1% of the image is left. Check image units.")

    # Fill internal missing values iteratively using a Gaussian filter
    fill_mask = np.isnan(t1map_data) & mask
    while fill_mask.sum() > 0:
        logger.info(f"Filling in {fill_mask.sum()} voxels within the mask.")
        t1map_data[fill_mask] = nan_filter_gaussian(t1map_data, 1.0)[fill_mask]
        fill_mask = np.isnan(t1map_data) & mask

    processed_T1map = MRIData(t1map_data, t1map_mri.affine)
    if output is not None:
        save_mri_data(processed_T1map, output, dtype=np.single)

    return processed_T1map


def mixed_t1map(
    SE_nii_path: Path, IR_nii_path: Path, meta_path: Path, T1_low: float, T1_high: float, output: Path | None = None
) -> nibabel.nifti1.Nifti1Image:
    """I/O wrapper to generate a T1 map from SE and IR acquisitions."""
    se_mri = load_mri_data(SE_nii_path, dtype=np.single)
    ir_mri = load_mri_data(IR_nii_path, dtype=np.single)
    meta = json.loads(meta_path.read_text())

    t1_volume = compute_mixed_t1_array(se_mri.data, ir_mri.data, meta, T1_low, T1_high)

    nii = nibabel.nifti1.Nifti1Image(t1_volume, ir_mri.affine)
    nii.set_sform(nii.affine, "scanner")
    nii.set_qform(nii.affine, "scanner")

    if output is not None:
        nibabel.nifti1.save(nii, output)

    return nii


def mixed_t1map_postprocessing(SE_nii_path: Path, T1_path: Path, output: Path | None = None) -> nibabel.nifti1.Nifti1Image:
    """I/O wrapper to mask out non-CSF areas from a Mixed T1 map based on SE signal."""
    t1map_nii = nibabel.nifti1.load(T1_path)
    se_mri = load_mri_data(SE_nii_path, np.single)

    mask = create_csf_mask(se_mri.data, use_li=True)
    mask = skimage.morphology.erosion(mask)

    masked_t1map = t1map_nii.get_fdata(dtype=np.single)
    masked_t1map[~mask] = np.nan
    masked_t1map_nii = nibabel.nifti1.Nifti1Image(masked_t1map, t1map_nii.affine, t1map_nii.header)

    if output is not None:
        nibabel.nifti1.save(masked_t1map_nii, output)

    return masked_t1map_nii


def looklocker_t1map(looklocker_input: Path, timestamps: Path, output: Path | None = None) -> MRIData:
    """I/O wrapper to generate a Look-Locker T1 map from a NIfTI file."""
    ll_mri = load_mri_data(looklocker_input, dtype=np.single)
    # Convert timestamps from milliseconds to seconds
    time_s = np.loadtxt(timestamps) / 1000.0

    t1map_array = compute_looklocker_t1_array(ll_mri.data, time_s)
    t1map_mri = MRIData(t1map_array.astype(np.single), ll_mri.affine)

    if output is not None:
        save_mri_data(t1map_mri, output, dtype=np.single)

    return t1map_mri


def compute_mixed_t1_array(se_data: np.ndarray, ir_data: np.ndarray, meta: dict, t1_low: float, t1_high: float) -> np.ndarray:
    """
    Computes a Mixed T1 array from Spin-Echo and Inversion-Recovery volumes using a lookup table.

    Args:
        se_data (np.ndarray): 3D numpy array of the Spin-Echo modulus data.
        ir_data (np.ndarray): 3D numpy array of the Inversion-Recovery corrected real data.
        meta (dict): Dictionary containing sequence parameters ('TR_SE', 'TI', 'TE', 'ETL').
        t1_low (float): Lower bound for T1 generation grid.
        t1_high (float): Upper bound for T1 generation grid.

    Returns:
        np.ndarray: Computed T1 map as a 3D float32 array.
    """
    nonzero_mask = se_data != 0
    f_data = np.nan * np.zeros_like(ir_data)
    f_data[nonzero_mask] = ir_data[nonzero_mask] / se_data[nonzero_mask]

    tr_se, ti, te, etl = meta["TR_SE"], meta["TI"], meta["TE"], meta["ETL"]
    f_curve, t1_grid = T1_lookup_table(tr_se, ti, te, etl, t1_low, t1_high)

    interpolator = scipy.interpolate.interp1d(f_curve, t1_grid, kind="nearest", bounds_error=False, fill_value=np.nan)
    return interpolator(f_data).astype(np.single)


def compute_hybrid_t1_array(ll_data: np.ndarray, mixed_data: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    """
    Creates a hybrid T1 array by selectively substituting Look-Locker voxels with Mixed voxels.

    Substitution occurs only if BOTH the Look-Locker AND Mixed T1 values exceed the threshold,
    AND the voxel falls within the provided CSF mask.

    Args:
        ll_data (np.ndarray): 3D numpy array of Look-Locker T1 values.
        mixed_data (np.ndarray): 3D numpy array of Mixed T1 values.
        mask (np.ndarray): 3D boolean mask (typically eroded CSF).
        threshold (float): T1 threshold value (in ms).

    Returns:
        np.ndarray: Hybrid 3D T1 array.
    """
    hybrid = ll_data.copy()
    newmask = mask & (ll_data > threshold) & (mixed_data > threshold)
    hybrid[newmask] = mixed_data[newmask]
    return hybrid


def hybrid_t1map(
    LL_path: Path, mixed_path: Path, csf_mask_path: Path, threshold: float, erode: int = 0, output: Path | None = None
) -> nibabel.nifti1.Nifti1Image:
    """I/O wrapper for merging a Look-Locker and a Mixed T1 map."""
    mixed_mri = nibabel.nifti1.load(mixed_path)
    ll_mri = nibabel.nifti1.load(LL_path)

    csf_mask_mri = nibabel.nifti1.load(csf_mask_path)
    csf_mask = csf_mask_mri.get_fdata().astype(bool)

    if erode > 0:
        csf_mask = skimage.morphology.erosion(csf_mask, skimage.morphology.ball(erode))

    hybrid = compute_hybrid_t1_array(ll_mri.get_fdata(), mixed_mri.get_fdata(), csf_mask, threshold)

    hybrid_nii = nibabel.nifti1.Nifti1Image(hybrid, affine=ll_mri.affine, header=ll_mri.header)
    if output is not None:
        nibabel.nifti1.save(hybrid_nii, output)

    return hybrid_nii
