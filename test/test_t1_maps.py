import numpy as np
import pytest

from mritk.t1_maps.t1_maps import remove_outliers, compute_mixed_t1_array, compute_hybrid_t1_array, create_largest_island_mask
from mritk.data.base import MRIData
from mritk.t1_maps.t1_to_r1 import compute_r1_array, convert_T1_to_R1, T1_to_R1


def test_compute_r1_array_standard():
    """Test basic T1 to R1 mathematical conversion."""
    t1_data = np.array([500.0, 1000.0, 2000.0])

    # Expected R1 = 1000 / T1
    expected = np.array([2.0, 1.0, 0.5])

    r1_data = compute_r1_array(t1_data, scale=1000.0)
    np.testing.assert_array_almost_equal(r1_data, expected)


def test_compute_r1_array_clipping():
    """Test that values outside the [t1_low, t1_high] bounds are safely set to NaN."""
    t1_data = np.array([0.5, 500.0, 6000.0, 10000.0])
    t1_low = 1.0
    t1_high = 5000.0

    r1_data = compute_r1_array(t1_data, scale=1000.0, t1_low=t1_low, t1_high=t1_high)

    # index 0 (0.5) < 1.0 -> NaN
    # index 1 (500) -> 2.0
    # index 2 (6000) > 5000.0 -> NaN
    # index 3 (10000) > 5000.0 -> NaN

    assert np.isnan(r1_data[0])
    assert r1_data[1] == 2.0
    assert np.isnan(r1_data[2])
    assert np.isnan(r1_data[3])


def test_convert_t1_to_r1_mridata():
    """Test the conversion properly preserves the MRIData class attributes (affine)."""
    t1_data = np.array([[[1000.0, 2000.0]]])
    affine = np.eye(4)
    mri = MRIData(data=t1_data, affine=affine)

    r1_mri = convert_T1_to_R1(mri, scale=1000.0)

    expected_r1 = np.array([[[1.0, 0.5]]])

    np.testing.assert_array_almost_equal(r1_mri.data, expected_r1)
    np.testing.assert_array_equal(r1_mri.affine, affine)


def test_t1_to_r1_invalid_input():
    """Test the wrapper function throws ValueError on an invalid type input."""
    with pytest.raises(ValueError, match="Input should be a Path or MRIData"):
        # Explicitly passing a raw string instead of Path/MRIData
        T1_to_R1(input_mri="not_a_path_or_mridata")


def test_remove_outliers():
    """Test that data is appropriately masked and clipped to physiological T1 bounds."""
    # 2x2x1 Mock Data
    data = np.array([[[10.0], [500.0]], [[1500.0], [8000.0]]])

    # Mask out the first element
    mask = np.array([[[False], [True]], [[True], [True]]])

    t1_low = 100.0
    t1_high = 2000.0

    result = remove_outliers(data, mask, t1_low, t1_high)

    # Expected:
    # [0,0,0] -> NaN (masked out)
    # [0,1,0] -> 500.0 (valid)
    # [1,0,0] -> 1500.0 (valid)
    # [1,1,0] -> NaN (exceeds t1_high)

    assert np.isnan(result[0, 0, 0])
    assert result[0, 1, 0] == 500.0
    assert result[1, 0, 0] == 1500.0
    assert np.isnan(result[1, 1, 0])


def test_compute_mixed_t1_array():
    """Test generating a T1 map from SE and IR modalities via interpolation."""
    se_data = np.array([[[1000.0, 1000.0]]])
    # IR signals at varying levels
    ir_data = np.array([[[-500.0, 500.0]]])

    meta = {"TR_SE": 1000.0, "TI": 100.0, "TE": 10.0, "ETL": 5}

    t1_low = 100.0
    t1_high = 3000.0

    t1_volume = compute_mixed_t1_array(se_data, ir_data, meta, t1_low, t1_high)

    # Should output same shape
    assert t1_volume.shape == (1, 1, 2)
    # T1 maps should not contain negative values in valid tissue
    assert np.all(t1_volume[~np.isnan(t1_volume)] > 0)


def test_compute_hybrid_t1_array():
    """Test hybrid array logic merges LL and Mixed appropriately based on threshold and mask."""
    # 1D array for simplicity (4 voxels)
    ll_data = np.array([1000.0, 2000.0, 1000.0, 2000.0])
    mixed_data = np.array([500.0, 500.0, 3000.0, 3000.0])

    # Voxel 3 is unmasked
    mask = np.array([True, True, True, False])
    threshold = 1500.0

    hybrid = compute_hybrid_t1_array(ll_data, mixed_data, mask, threshold)

    # Evaluation: Substitution happens ONLY if BOTH > threshold AND inside mask.
    # Voxel 0: 1000 < 1500 -> Keep LL (1000.0)
    # Voxel 1: Mixed 500 < 1500 -> Keep LL (2000.0)
    # Voxel 2: LL (1000) < 1500 -> Keep LL (1000.0) ... wait, let's fix ll_data[2] to test properly
    # Let's run it as-is:
    assert hybrid[0] == 1000.0
    assert hybrid[1] == 2000.0
    assert hybrid[2] == 1000.0
    assert hybrid[3] == 2000.0  # Unmasked, so keep LL

    # Let's explicitly trigger the merge condition
    ll_data[2] = 2000.0
    hybrid2 = compute_hybrid_t1_array(ll_data, mixed_data, mask, threshold)
    # Voxel 2: LL(2000) > 1500 AND Mixed(3000) > 1500 AND Mask=True -> Merge!
    assert hybrid2[2] == 3000.0


def test_create_largest_island_mask():
    """Test morphology logic identifies the primary body of data and ignores disconnected noise."""
    # Create a 15x15x15 empty space (3375 voxels, which is > 1000 so the background isn't
    # accidentally filled in by remove_small_holes)
    data = np.full((15, 15, 15), np.nan)

    # Create a large block in the center (Island 1)
    data[5:10, 5:10, 5:10] = 100.0

    # Create a tiny disconnected speck in the corner (Island 2)
    data[0, 0, 0] = 50.0

    # Run with small morphology radiuses
    mask = create_largest_island_mask(data, radius=1, erode_dilate_factor=1.0)

    # Speck should be dropped, major block should be True
    assert mask[0, 0, 0] == np.False_
    assert mask[7, 7, 7] == np.True_
