# MRI DICOM to NIfTI conversion Module

# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


import shutil
import subprocess
import tempfile
import logging
import json
from pathlib import Path
from typing import Optional

import nibabel
import numpy as np

from ..data.io import load_mri_data, save_mri_data
from ..t1_maps.utils import VOLUME_LABELS, read_dicom_trigger_times
from .utils import extract_single_volume

logger = logging.getLogger(__name__)


def _extract_frame_metadata(frame_fg) -> dict:
    """
    Extracts core physical parameters (TR, TE, TI, ETL) from a DICOM frame functional group.

    Args:
        frame_fg: The PerFrameFunctionalGroupsSequence element for a specific frame.

    Returns:
        dict: A dictionary containing available MR timing parameters.
    """
    descrip = {
        "TR": float(frame_fg.MRTimingAndRelatedParametersSequence[0].RepetitionTime),
        "TE": float(frame_fg.MREchoSequence[0].EffectiveEchoTime),
    }

    if hasattr(frame_fg.MRModifierSequence[0], "InversionTimes"):
        descrip["TI"] = frame_fg.MRModifierSequence[0].InversionTimes[0]

    if hasattr(frame_fg.MRTimingAndRelatedParametersSequence[0], "EchoTrainLength"):
        descrip["ETL"] = frame_fg.MRTimingAndRelatedParametersSequence[0].EchoTrainLength

    return descrip


import shlex
import logging

logger = logging.getLogger(__name__)


def run_dcm2niix(input_path: Path, output_dir: Path, form: str, extra_args: str = "", check: bool = True):
    """
    Utility wrapper to execute the dcm2niix command-line tool securely.

    Args:
        input_path (Path): Path to the input DICOM file/folder.
        output_dir (Path): Path to the target output directory.
        form (str): Output filename format string.
        extra_args (str, optional): Additional command line arguments. Defaults to "".
        check (bool, optional): If True, raises an exception on failure. Defaults to True.

    Raises:
        RuntimeError: If the dcm2niix executable is not found in the system PATH.
        subprocess.CalledProcessError: If the command fails and `check` is True.
    """
    # 1. Locate the executable securely
    executable = shutil.which("dcm2niix")
    if executable is None:
        raise RuntimeError(
            "The 'dcm2niix' executable was not found. Please ensure it is installed and available in your system PATH."
        )

    # 2. Build the arguments list safely
    args = [executable, "-f", form]

    # Safely parse the extra string arguments into a list
    if extra_args:
        args.extend(shlex.split(extra_args))

    args.extend(["-o", str(output_dir), str(input_path)])

    # Reconstruct the command string purely for logging purposes
    cmd_str = shlex.join(args)
    logger.debug(f"Executing: {cmd_str}")

    try:
        # 3. Execute without shell=True for better security and stability
        subprocess.run(args, check=check, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"dcm2niix execution failed.\nCommand: {cmd_str}\nError: {e.stderr}")
        if check:
            raise


def extract_mixed_dicom(dcmpath: Path, subvolumes: list[str]) -> list[dict]:
    """
    Reads a Mixed DICOM file and splits it into independent NIfTI subvolumes.

    Args:
        dcmpath (Path): Path to the input DICOM file.
        subvolumes (list[str]): List of volume labels mapping to the slices in the DICOM.

    Returns:
        list[dict]: A list containing dictionaries with a generated 'nifti' image
        and a 'descrip' metadata dictionary for each requested subvolume.
    """
    import pydicom

    dcm = pydicom.dcmread(str(dcmpath))
    frames_total = int(dcm.NumberOfFrames)

    # [0x2001, 0x1018] is a private Philips tag representing 'Number of Slices MR'
    frames_per_volume = dcm[0x2001, 0x1018].value
    num_volumes = frames_total // frames_per_volume
    assert num_volumes * frames_per_volume == frames_total, "Subvolume dimensions do not evenly divide the total frames."

    pixel_data = dcm.pixel_array.astype(np.single)
    frame_fg_sequence = dcm.PerFrameFunctionalGroupsSequence

    vols_out = []
    for volname in subvolumes:
        vol_idx = VOLUME_LABELS.index(volname)

        # Find volume slices representing the current subvolume
        subvol_idx_start = vol_idx * frames_per_volume
        subvol_idx_end = (vol_idx + 1) * frames_per_volume
        frame_fg = frame_fg_sequence[subvol_idx_start]

        logger.info(
            f"Converting volume {vol_idx + 1}/{len(VOLUME_LABELS)}: '{volname}' "
            f"between indices {subvol_idx_start}-{subvol_idx_end} out of {frames_total}."
        )

        mri = extract_single_volume(pixel_data[subvol_idx_start:subvol_idx_end], frame_fg)

        nii_oriented = nibabel.nifti1.Nifti1Image(mri.data, mri.affine)
        nii_oriented.set_sform(nii_oriented.affine, "scanner")
        nii_oriented.set_qform(nii_oriented.affine, "scanner")

        description = _extract_frame_metadata(frame_fg)
        vols_out.append({"nifti": nii_oriented, "descrip": description})

    return vols_out


def dicom_to_looklocker(dicomfile: Path, outpath: Path):
    """
    Converts a Look-Locker DICOM file to a standardized NIfTI format.

    Extracts trigger times to a sidecar text file, delegates conversion to dcm2niix,
    and standardizes the output type to single-precision float (intent_code=2001).

    Args:
        dicomfile (Path): Path to the input DICOM file.
        outpath (Path): Desired output path for the converted .nii.gz file.
    """
    outdir, form = outpath.parent, outpath.stem
    outdir.mkdir(exist_ok=True, parents=True)

    # Extract and save trigger times
    times = read_dicom_trigger_times(dicomfile)
    np.savetxt(outdir / f"{form}_trigger_times.txt", times)

    with tempfile.TemporaryDirectory(prefix=outpath.stem) as tmpdir:
        tmppath = Path(tmpdir)

        # Delegate heavy lifting to dcm2niix
        run_dcm2niix(dicomfile, tmppath, form, extra_args="-z y --ignore_trigger_times", check=True)

        # Copy metadata sidecar
        shutil.copy(tmppath / f"{form}.json", outpath.with_suffix(".json"))

        # Reload and save to standardize intent codes and precision
        mri = load_mri_data(tmppath / f"{form}.nii.gz", dtype=np.double)
        save_mri_data(mri, outpath.with_suffix(".nii.gz"), dtype=np.single, intent_code=2001)


def dicom_to_mixed(dcmpath: Path, outpath: Path, subvolumes: Optional[list[str]] = None):
    """
    Converts a Mixed sequence DICOM file into independent subvolume NIfTIs.

    Generates dedicated images for Spin-Echo, Inversion-Recovery, etc.,
    and saves sequence timing metadata to a JSON sidecar.

    Args:
        dcmpath (Path): Path to the input Mixed DICOM file.
        outpath (Path): Base path for output files. Suffixes are automatically appended.
        subvolumes (list[str], optional): specific subvolumes to extract.
            Defaults to all known VOLUME_LABELS.
    """
    subvolumes = subvolumes or VOLUME_LABELS
    assert all([volname in VOLUME_LABELS for volname in subvolumes]), (
        f"Invalid subvolume name in {subvolumes}, must be one of {VOLUME_LABELS}"
    )

    outdir, form = outpath.parent, outpath.stem
    outdir.mkdir(exist_ok=True, parents=True)

    vols = extract_mixed_dicom(dcmpath, subvolumes)
    meta = {}

    for vol, volname in zip(vols, subvolumes):
        output = outpath.with_name(f"{outpath.stem}_{volname}.nii.gz")
        nibabel.nifti1.save(vol["nifti"], output)

        descrip = vol["descrip"]
        try:
            if volname == "SE-modulus":
                meta["TR_SE"] = descrip["TR"]
                meta["TE"] = descrip["TE"]
                meta["ETL"] = descrip["ETL"]
            elif volname == "IR-corrected-real":
                meta["TR_IR"] = descrip["TR"]
                meta["TI"] = descrip["TI"]
        except KeyError as e:
            logger.error(f"Missing required metadata for {volname}: {descrip}")
            raise e

    # Write merged metadata sidecar
    (outdir / f"{form}_meta.json").write_text(json.dumps(meta, indent=4))

    # Attempt standard dcm2niix conversion (soft failure allowed for legacy behavior)
    run_dcm2niix(dcmpath, outdir, form, extra_args="-w 0 --terse -b o", check=False)
