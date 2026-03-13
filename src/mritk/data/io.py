# MRI Data IO Module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


from pathlib import Path
import nibabel
import numpy as np
import numpy.typing as npt
from typing import Optional


def check_suffix(filepath: Path):
    suffix = filepath.suffix
    if suffix == ".gz":
        suffixes = filepath.suffixes
        if len(suffixes) >= 2 and suffixes[-2] == ".nii":
            return ".nii.gz"
    return suffix


def load_mri_data(filepath: Path, dtype: type = np.float64) -> tuple[np.ndarray, np.ndarray]:
    suffix = check_suffix(filepath)
    if suffix in (".nii", ".nii.gz"):
        mri = nibabel.nifti1.load(filepath)
    elif suffix in (".mgz", ".mgh"):
        mri = nibabel.freesurfer.mghformat.load(filepath)
    else:
        raise ValueError(f"Invalid suffix {filepath}, should be either '.nii', or '.mgz'")

    affine = mri.affine
    if affine is None:
        raise RuntimeError("MRI does not contain an affine")

    data = np.asarray(mri.get_fdata("unchanged"), dtype=dtype)

    return data, affine


def save_mri_data(data: np.ndarray, affine: np.ndarray, save_path: Path, dtype: npt.DTypeLike, intent_code: Optional[int] = None):
    suffix = check_suffix(save_path)
    if suffix in (".nii", ".nii.gz"):
        nii = nibabel.nifti1.Nifti1Image(data.astype(dtype), affine)
        if intent_code is not None:
            nii.header.set_intent(intent_code)
        nibabel.nifti1.save(nii, save_path)
    elif suffix in (".mgz", ".mgh"):
        mgh = nibabel.freesurfer.mghformat.MGHImage(data.astype(dtype), affine)
        if intent_code is not None:
            mgh.header.set_intent(intent_code)
        nibabel.freesurfer.mghformat.save(mgh, save_path)
    else:
        raise ValueError(f"Invalid suffix {save_path}, should be either '.nii', or '.mgz'")
