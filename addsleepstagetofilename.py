from pathlib import Path
import re
import shutil

import pandas as pd


# ==========================================================
# USER SETTINGS
# ==========================================================

PROJECT_DIR = Path("/media/brainlab-uwo/Elements1/Kayla")

# Folder containing the BIDS-style fMRI files.
FMRI_DIR = PROJECT_DIR / "AWS_EEG_FMRI_DATA_KAYLA"

# Folder containing YASA_RESULTS/sub-*/ses-*/*/hypnogram.csv.
YASA_DIR = PROJECT_DIR / "YASA_RESULTS"

# Optional: use a manually prepared CSV instead of building one from YASA.
# Required columns for a manual CSV: subject, session, acq, stage
STAGE_CSV = None

# False = make renamed copies, safer.
# True  = actually rename original files.
RENAME_ORIGINALS = False

# If RENAME_ORIGINALS = False, copied files go here.
OUTPUT_DIR = PROJECT_DIR / "renamed_fmri_files"

# If using YASA hypnograms, label each fMRI run by this summary of its stages.
# Current option: "dominant" = most frequent stage in that recording.
STAGE_SUMMARY_METHOD = "dominant"


# ==========================================================
# STAGE NAME CONVERSION
# ==========================================================

STAGE_TO_SUFFIX = {
    "wake": "W",
    "awake": "W",
    "w": "W",

    "rem": "R",
    "r": "R",

    "n1": "N1",
    "nrem1": "N1",
    "nrem 1": "N1",

    "n2": "N2",
    "nrem2": "N2",
    "nrem 2": "N2",

    "n3": "N3",
    "nrem3": "N3",
    "nrem 3": "N3",
    "slow wave": "N3",
    "sws": "N3",

    "art": "ART",
    "artifact": "ART",
    "uns": "UNS",
    "unknown": "UNS",
}

STAGE_SUFFIXES = tuple(f"_{suffix}" for suffix in sorted(set(STAGE_TO_SUFFIX.values())))


def normalize_stage(stage):
    stage_clean = str(stage).strip().lower()

    if stage_clean not in STAGE_TO_SUFFIX:
        raise ValueError(
            f"Unknown sleep stage: '{stage}'. "
            f"Allowed stages are: {sorted(STAGE_TO_SUFFIX.keys())}"
        )

    return STAGE_TO_SUFFIX[stage_clean]


def split_nii_filename(path):
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7], ".nii.gz"

    if name.endswith(".nii"):
        return name[:-4], ".nii"

    raise ValueError(f"Not a NIfTI file: {name}")


def extract_bids_entity(text, key):
    match = re.search(rf"(?:^|_){key}-([^_]+)", text)
    return match.group(1) if match else None


def recording_to_mapping_row(hypnogram_path):
    df = pd.read_csv(hypnogram_path)

    required = {"Subject", "Session", "Recording", "Stage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{hypnogram_path} is missing columns: {missing}")

    if STAGE_SUMMARY_METHOD != "dominant":
        raise ValueError(f"Unsupported STAGE_SUMMARY_METHOD: {STAGE_SUMMARY_METHOD}")

    stage_counts = df["Stage"].dropna().astype(str).str.strip().value_counts()
    if stage_counts.empty:
        raise ValueError(f"{hypnogram_path} has no sleep stages.")

    recording = str(df["Recording"].iloc[0])
    acq = extract_bids_entity(recording, "acq")
    if acq is None:
        raise ValueError(f"Could not extract acq-* from recording name: {recording}")

    dominant_stage = stage_counts.index[0]

    return {
        "subject": str(df["Subject"].iloc[0]).strip(),
        "session": str(df["Session"].iloc[0]).strip(),
        "acq": acq,
        "stage": dominant_stage,
        "n_epochs": int(len(df)),
        "stage_counts": ";".join(f"{stage}:{count}" for stage, count in stage_counts.items()),
        "source_hypnogram": str(hypnogram_path),
    }


def build_stage_mapping_from_yasa():
    rows = []

    for hypnogram_path in sorted(YASA_DIR.glob("sub-*/ses-*/*/hypnogram.csv")):
        rows.append(recording_to_mapping_row(hypnogram_path))

    if not rows:
        raise FileNotFoundError(f"No per-recording hypnogram.csv files found under {YASA_DIR}")

    return pd.DataFrame(rows)


def load_stage_mapping():
    if STAGE_CSV is not None:
        stage_csv = Path(STAGE_CSV)
        if not stage_csv.exists():
            raise FileNotFoundError(f"STAGE_CSV does not exist: {stage_csv}")
        df = pd.read_csv(stage_csv)
    else:
        df = build_stage_mapping_from_yasa()
        mapping_path = OUTPUT_DIR / "stage_mapping_from_yasa.csv"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(mapping_path, index=False)
        print(f"Saved stage mapping to: {mapping_path}")

    required_columns = {"subject", "session", "acq", "stage"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Stage mapping is missing columns: {missing}. "
            f"Required columns are: {sorted(required_columns)}"
        )

    return df


def find_sleep_bold_files():
    return sorted(
        list(FMRI_DIR.rglob("*_task-sleep_*_bold.nii"))
        + list(FMRI_DIR.rglob("*_task-sleep_*_bold.nii.gz"))
    )


def find_matching_fmri(subject, session, acq, fmri_files):
    matches = []

    expected_parts = [
        f"{subject}_",
        f"{session}_",
        "_task-sleep_",
        f"_acq-{acq}_",
        "_bold.",
    ]

    for file in fmri_files:
        filename = file.name
        if all(part in filename for part in expected_parts):
            matches.append(file)

    return matches


def make_output_path(old_path, new_name):
    if RENAME_ORIGINALS:
        return old_path.with_name(new_name)

    relative_parent = old_path.parent.relative_to(FMRI_DIR)
    out_dir = OUTPUT_DIR / relative_parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / new_name


def main():
    if not FMRI_DIR.exists():
        raise FileNotFoundError(f"FMRI_DIR does not exist: {FMRI_DIR}")

    if not RENAME_ORIGINALS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_stage_mapping()
    fmri_files = find_sleep_bold_files()

    print(f"Found {len(fmri_files)} sleep BOLD fMRI files.")
    print(f"Found {len(df)} stage mapping rows.")

    rename_log = []

    for _, row in df.iterrows():
        subject = str(row["subject"]).strip()
        session = str(row["session"]).strip()
        acq = str(row["acq"]).strip()
        stage = row["stage"]

        suffix = normalize_stage(stage)
        matches = find_matching_fmri(subject, session, acq, fmri_files)

        if len(matches) == 0:
            print(f"[WARNING] No sleep BOLD file found for {subject}, {session}, acq-{acq}")
            continue

        if len(matches) > 1:
            print(f"[WARNING] Multiple sleep BOLD files found for {subject}, {session}, acq-{acq}:")
            for match in matches:
                print(f"    {match}")
            print("Skipping this entry to avoid renaming the wrong file.")
            continue

        old_path = matches[0]
        stem, ext = split_nii_filename(old_path)

        if stem.endswith(STAGE_SUFFIXES):
            print(f"[SKIP] Already has sleep-stage suffix: {old_path.name}")
            continue

        new_name = f"{stem}_{suffix}{ext}"
        new_path = make_output_path(old_path, new_name)

        if new_path.exists():
            print(f"[SKIP] Output already exists: {new_path}")
            continue

        if RENAME_ORIGINALS:
            old_path.rename(new_path)
            action = "RENAMED"
        else:
            shutil.copy2(old_path, new_path)
            action = "COPIED"

        print(f"[{action}] {old_path.name} -> {new_path.name}")

        rename_log.append({
            "subject": subject,
            "session": session,
            "acq": acq,
            "stage": stage,
            "suffix": suffix,
            "old_file": str(old_path),
            "new_file": str(new_path),
            "action": action,
        })

    log_df = pd.DataFrame(rename_log)

    if not log_df.empty:
        log_path = (OUTPUT_DIR if not RENAME_ORIGINALS else FMRI_DIR) / "rename_log.csv"
        log_df.to_csv(log_path, index=False)
        print(f"\nSaved rename log to: {log_path}")
    else:
        print("\nNo files were copied or renamed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
