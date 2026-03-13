# MRI Data Base class and functions Module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


import numpy as np
from pathlib import Path
from .io import load_mri_data, save_mri_data


class MRIData:
    def __init__(self, data: np.ndarray, affine: np.ndarray):
        self.data = data
        self.affine = affine
        self.dtype = data.dtype

    @classmethod
    def from_file(cls, filepath: Path):
        data, affine = load_mri_data(filepath, np.float64)
        return cls(data=data, affine=affine)

    def save_mri_data(self, save_path: Path, intent_code: int | None = None):
        save_mri_data(self.data, self.affine, save_path, self.dtype, intent_code)

    def get_data(self):
        return self.data

    def get_metadata(self):
        return self.affine

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def voxel_ml_volume(self) -> float:
        # Calculate the volume of a single voxel in milliliters
        voxel_volume_mm3 = abs(np.linalg.det(self.affine[:3, :3]))
        voxel_volume_ml = voxel_volume_mm3 / 1000.0  # Convert from mm^3 to ml
        return voxel_volume_ml
