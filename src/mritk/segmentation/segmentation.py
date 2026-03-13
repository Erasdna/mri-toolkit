from ..data.base import MRIData
import numpy as np
import pandas as pd
from .lookup_table import read_lut
from pathlib import Path
from ..data.io import load_mri_data


class Segmentation(MRIData):
    def __init__(self, data: np.ndarray, affine: np.ndarray, lut: pd.DataFrame | None = None):
        super().__init__(data, affine)
        self.data = self.data.astype(int)
        self.rois = np.unique(self.data[self.data > 0])
        if lut is not None:
            self.lut = lut
        else:
            self.lut = pd.DataFrame({"Label": self.rois}, index=self.rois)

    @classmethod
    def from_file(cls, filepath: Path, dtype: type = int):
        data, affine = load_mri_data(filepath, dtype=dtype)
        return cls(data=data, affine=affine)

    @property
    def num_rois(self) -> int:
        return len(self.rois)

    @property
    def roi_labels(self) -> np.ndarray:
        return self.rois

    def get_roi_labels(self, rois: np.ndarray[int] | None = None) -> pd.DataFrame:
        if rois is None:
            rois = self.rois

        if not np.isin(rois, self.rois).all():
            raise ValueError("Some of the provided ROIs are not present in the segmentation.")

        return self.lut.loc[self.lut.index.isin(rois), ["Label"]].rename_axis("ROI").reset_index()


class FreeSurferSegmentation(Segmentation):
    def __init__(self, data: np.ndarray, affine: np.ndarray):
        # Retrieve the FreeSurfer LUT, which contains the labels associated with each ROI
        lut = read_lut(None)  # Finds freesurfer LUT
        super().__init__(data, affine, lut)

        # TODO: verify that all labels in the segmentation are valid FS labels


class ExtendedFreeSurferSegmentation(FreeSurferSegmentation):
    def __init__(self, data: np.ndarray, affine: np.ndarray):
        super().__init__(data, affine)

    def get_roi_labels(self, rois: np.ndarray[int] | None = None) -> pd.DataFrame:
        rois = self.rois if rois is None else rois

        freesurfer_labels = super().get_roi_labels(rois % 10000).rename(columns={"ROI": "FreeSurfer_ROI"})
        tissue_type = self.get_tissue_type(rois)
        return freesurfer_labels.merge(
            tissue_type,
            left_on="FreeSurfer_ROI",
            right_on="FreeSurfer_ROI",
            how="outer",
        ).drop(columns=["FreeSurfer_ROI"])[["ROI", "Label", "tissue_type"]]

    def get_tissue_type(self, rois: np.ndarray[int] | None = None) -> pd.DataFrame:
        rois = self.rois if rois is None else rois
        tissue_types = pd.Series(
            data=np.where(rois < 10000, "Parenchyma", np.where(rois < 20000, "CSF", "Dura")),
            index=rois,
            name="tissue_type",
        )
        ret = pd.DataFrame(tissue_types, columns=["tissue_type"]).rename_axis("ROI").reset_index()
        ret["FreeSurfer_ROI"] = ret["ROI"] % 10000
        return ret
