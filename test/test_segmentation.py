from mritk.segmentation.segmentation import Segmentation, FreeSurferSegmentation, ExtendedFreeSurferSegmentation
from pathlib import Path
import numpy as np


def test_segmentation_initialization(example_segmentation: Segmentation):
    assert example_segmentation.data.shape == (100, 4)
    assert example_segmentation.affine.shape == (4, 4)
    assert example_segmentation.num_rois == 3
    assert set(example_segmentation.roi_labels) == {1, 2, 3}
    assert example_segmentation.lut.shape == (3, 1)
    assert set(example_segmentation.lut.columns) == {"Label"}


def test_freesurfer_segmentation_labels(mri_data_dir: Path):
    fs_seg = FreeSurferSegmentation.from_file(
        mri_data_dir
        / "mri-processed/mri_processed_data/sub-01\
            /segmentations/sub-01_seg-aparc+aseg_refined.nii.gz"
    )

    labels = fs_seg.get_roi_labels()
    assert not labels.empty
    assert set(labels["ROI"]) == set(fs_seg.roi_labels)


def test_extended_freesurfer_segmentation_labels(example_segmentation: Segmentation, mri_data_dir: Path):
    data = example_segmentation.data
    data[0:2, 0:2] = 10001  # csf
    data[3:5, 3:5] = 20001  # dura

    ext_fs_seg = ExtendedFreeSurferSegmentation(data, affine=np.eye(4))
    labels = ext_fs_seg.get_roi_labels()

    assert set(labels["ROI"]) == set(ext_fs_seg.roi_labels)
    assert labels.loc[labels["ROI"] == 10001, "tissue_type"].iloc[0] == "CSF"
    assert labels.loc[labels["ROI"] == 20001, "tissue_type"].iloc[0] == "Dura"
    assert labels.loc[labels["ROI"] == 10001, "Label"].iloc[0] == labels.loc[labels["ROI"] == 1, "Label"].iloc[0]
    assert labels.loc[labels["ROI"] == 20001, "Label"].iloc[0] == labels.loc[labels["ROI"] == 1, "Label"].iloc[0]
