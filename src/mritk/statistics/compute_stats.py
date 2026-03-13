# MRI Statistics Module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory

from typing import Optional
import numpy as np
import pandas as pd
import tqdm.rich

from ..data.orientation import assert_same_space
from .utils import prepend_info
from ..segmentation.segmentation import Segmentation
from ..data.base import MRIData
from .stat_functions import Statistic, Mean, Std, Median


def generate_stats_dataframe(
    seg: Segmentation,
    mri: MRIData,
    qois: list[Statistic] = [Mean, Std, Median],
    metadata: Optional[dict] = None,
) -> pd.DataFrame:
    # Verify that segmentation and MRI are in the same space
    assert_same_space(seg, mri)

    qoi_records = []  # Collects records related to qois
    roi_records = []  # Collects records related to ROIs,

    # Mask infinite values
    finite_mask = np.isfinite(mri.data)
    for roi in tqdm.rich.tqdm(seg.roi_labels, total=len(seg.roi_labels)):
        # Identify rois in segmentation
        region_mask = (seg.data == roi) * finite_mask
        # print(region_mask.shape)
        region_data = mri.data[region_mask]
        nb_nans = np.isnan(region_data).sum()

        voxelcount = len(region_data)

        roi_records.append(
            {
                "ROI": roi,
                "voxel_count": voxelcount,
                "volume_ml": seg.voxel_ml_volume * voxelcount,
                "num_nan_values": nb_nans,
            }
        )
        # Iterate qoi functions
        for qoi in qois:
            qoi_value = qoi(region_data)
            # Store the qoi value in a dataframe, along with the roi label and description
            qoi_records.append({"ROI": roi, "statistic": qoi.name, "value": qoi_value})

    df = pd.DataFrame.from_records(qoi_records)
    df_roi = pd.DataFrame.from_records(roi_records)
    df = df.merge(df_roi, on="ROI", how="left")

    # Add some metadata to each row
    if metadata is not None:
        df = prepend_info(df, **(metadata))
    return df
