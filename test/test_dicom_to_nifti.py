from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np

from mritk.t1_maps.dicom_to_nifti import (
    _extract_frame_metadata,
    run_dcm2niix,
    extract_mixed_dicom,
    VOLUME_LABELS,
)


def test_extract_frame_metadata():
    """Test the extraction of relevant MR metadata parameters from DICOM tags."""
    # Mocking a DICOM Functional Group hierarchy
    mock_frame = MagicMock()
    mock_frame.MRTimingAndRelatedParametersSequence[0].RepetitionTime = 1500.0
    mock_frame.MREchoSequence[0].EffectiveEchoTime = 10.0
    mock_frame.MRModifierSequence[0].InversionTimes = [150.0]
    mock_frame.MRTimingAndRelatedParametersSequence[0].EchoTrainLength = 5

    meta = _extract_frame_metadata(mock_frame)

    assert meta["TR"] == 1500.0
    assert meta["TE"] == 10.0
    assert meta["TI"] == 150.0
    assert meta["ETL"] == 5


@patch("subprocess.run")
def test_run_dcm2niix(mock_run):
    """Test that the dcm2niix command constructor triggers properly."""
    input_path = Path("/input/data.dcm")
    output_dir = Path("/output/")

    # Test valid execution
    run_dcm2niix(input_path, output_dir, form="test_form", extra_args="-z y")

    # Verify the constructed shell command
    mock_run.assert_called_once()
    args, _ = mock_run.call_args
    cmd = args[0]

    assert "dcm2niix" in cmd[0]
    assert "test_form" in cmd
    assert "-z" in cmd
    assert "y" in cmd


@patch("mritk.t1_maps.dicom_to_nifti.extract_single_volume")
@patch("pydicom.dcmread")
def test_extract_mixed_dicom(mock_dcmread, mock_extract_single):
    """Test parsing a multi-volume DICOM file into independent subvolumes."""
    # Mocking the pydicom output
    mock_dcm = MagicMock()
    mock_dcm.NumberOfFrames = 20
    # Private tag for "Number of slices MR"
    mock_slice_tag = MagicMock()
    mock_slice_tag.value = 10

    # We have to mock __getitem__ because it's called via dcm[0x2001, 0x1018]
    def getitem_side_effect(key):
        if key == (0x2001, 0x1018):
            return mock_slice_tag
        return MagicMock()

    mock_dcm.__getitem__.side_effect = getitem_side_effect

    # Dummy pixel array
    mock_dcm.pixel_array = np.zeros((20, 2, 2))

    # Mocking Frame metadata sequences
    mock_frame_fg = MagicMock()
    mock_frame_fg.MRTimingAndRelatedParametersSequence[0].RepetitionTime = 1000.0
    mock_frame_fg.MREchoSequence[0].EffectiveEchoTime = 5.0

    # List of 20 frames
    mock_dcm.PerFrameFunctionalGroupsSequence = [mock_frame_fg] * 20
    mock_dcmread.return_value = mock_dcm

    # Mock the volume extraction output
    mock_mri_data = MagicMock()
    mock_mri_data.data = np.ones((10, 2, 2))
    mock_mri_data.affine = np.eye(4)
    mock_extract_single.return_value = mock_mri_data

    # Run the function requesting just the first two volumes
    dcmpath = Path("/dummy/file.dcm")
    test_subvolumes = [VOLUME_LABELS[0], VOLUME_LABELS[1]]

    results = extract_mixed_dicom(dcmpath, test_subvolumes)

    # Verifications
    assert len(results) == 2
    assert "nifti" in results[0]
    assert "descrip" in results[0]
    assert results[0]["descrip"]["TR"] == 1000.0

    # Ensure extract_single_volume was called twice (once for each subvolume)
    assert mock_extract_single.call_count == 2
