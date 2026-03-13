"""Concentration maps module

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

from pathlib import Path
from typing import Optional

import numpy as np
from ..data.base import MRIData
from ..data.orientation import assert_same_space


def concentration_from_T1(T1: np.ndarray, T1_0: np.ndarray, r1: float) -> np.ndarray:
    C = 1 / r1 * (1 / T1 - 1 / T1_0)
    return C


def concentration_from_R1(R1: np.ndarray, R1_0: np.ndarray, r1: float) -> np.ndarray:
    C = 1 / r1 * (R1 - R1_0)
    return C


def concentration(
    T1_mri: MRIData,
    T10_mri: MRIData,
    output: Optional[Path] = None,
    r1: float = 0.0045,
    mask: Optional[MRIData] = None,
) -> MRIData:
    # T1_mri = load_mri_data(input, np.single)
    # T10_mri = load_mri_data(reference, np.single)
    assert_same_space(T1_mri, T10_mri)

    if mask is not None:
        assert_same_space(mask, T10_mri)
        mask_data = mask.data * (T10_mri.data > 1e-10) * (T1_mri.data > 1e-10)
        T1_mri.data *= mask_data
        T10_mri.data *= mask_data
    else:
        mask_data = (T10_mri.data > 1e-10) * (T1_mri.data > 1e-10)
        T1_mri.data[~mask_data] = np.nan
        T10_mri.data[~mask_data] = np.nan

    concentrations = np.nan * np.zeros_like(T10_mri.data)
    concentrations[mask_data] = concentration_from_T1(T1=T1_mri.data[mask_data], T1_0=T10_mri.data[mask_data], r1=r1)
    mri_data = MRIData(data=concentrations, affine=T10_mri.affine)
    if output is not None:
        mri_data.save_mri_data(output)
    return mri_data
