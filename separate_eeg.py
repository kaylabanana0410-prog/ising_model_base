import os
import glob
import mne
import yasa
import pandas as pd
import matplotlib.pyplot as plt


ROOT = "AWS_EEG_FMRI_DATA_KAYLA"
OUTDIR = "YASA_RESULTS"


def get_brainvision_data_file(vhdr_file):
    with open(vhdr_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("DataFile="):
                data_file = line.split("=", 1)[1].strip()
                return os.path.join(os.path.dirname(vhdr_file), data_file)
    return None


def add_bipolar_channel(raw, anode, cathode, name):
    if name in raw.ch_names:
        return

    if anode not in raw.ch_names or cathode not in raw.ch_names:
        return

    mne.set_bipolar_reference(
        raw,
        anode=anode,
        cathode=cathode,
        ch_name=name,
        drop_refs=False,
        copy=False,
        verbose=False,
    )


def plot_full_hypnogram(df, out_path, title):
    stage_to_num = {
        "N3": 0,
        "N2": 1,
        "N1": 2,
        "REM": 3,
        "WAKE": 4,
        "W": 4,
        "ART": -1,
        "UNS": -1,
    }
    y = df["Stage"].map(stage_to_num)
    x = df["PatientEpoch"]

    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    ax.step(x, y, where="post", color="black", linewidth=1.2)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["N3", "N2", "N1", "REM", "WAKE"])
    ax.set_xlabel("30-second epoch")
    ax.set_ylabel("Sleep stage")
    ax.set_title(title)
    ax.set_ylim(-0.5, 4.5)
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


os.makedirs(OUTDIR, exist_ok=True)
patient_results = {}

# Find all EEG BrainVision files
vhdr_files = glob.glob(
    os.path.join(ROOT, "sub-*", "ses-*", "eeg", "*.vhdr")
)

print(f"Found {len(vhdr_files)} EEG recordings")


for vhdr_file in vhdr_files:

    try:
        print("\n--------------------------------")
        print("Processing:")
        print(vhdr_file)

        # Get subject/session/file name
        parts = vhdr_file.split(os.sep)

        subject = parts[-4]
        session = parts[-3]
        recording = os.path.basename(vhdr_file).replace(".vhdr", "")

        data_file = get_brainvision_data_file(vhdr_file)
        if data_file is None:
            print("No DataFile entry found in header, skipping")
            continue

        if not os.path.exists(data_file):
            print("Missing EEG binary file, skipping:")
            print(data_file)
            continue

        save_dir = os.path.join(
            OUTDIR,
            subject,
            session,
            recording,
        )

        os.makedirs(save_dir, exist_ok=True)

        # -------------------------
        # Load EEG
        # -------------------------
        raw = mne.io.read_raw_brainvision(
            vhdr_file,
            preload=True,
            eog=["E1", "E2"],
            misc=["ECG", "EMG"],
            verbose=False,
        )

        # -------------------------
        # Find channels
        # -------------------------
        print("Channels:")
        print(raw.ch_names)

        add_bipolar_channel(raw, "C3", "TP10", "C3-TP10")
        add_bipolar_channel(raw, "C4", "TP9", "C4-TP9")
        add_bipolar_channel(raw, "E1", "E2", "E1-E2")

        # EEG
        eeg_candidates = [
            "C3-TP10",
            "C4-TP9",
            "C3",
            "C4",
            "C3-M2",
            "C4-M1",
        ]

        eeg_channel = None

        for ch in eeg_candidates:
            if ch in raw.ch_names:
                eeg_channel = ch
                break

        if eeg_channel is None:
            print("No EEG channel found, skipping")
            continue

        # EOG
        eog_channels = [
            ch for ch in ["E1-E2", "E1", "E2"]
            if ch in raw.ch_names
        ]

        # EMG
        emg_channel = None
        if "EMG" in raw.ch_names:
            emg_channel = "EMG"

        print("Using:")
        print("EEG:", eeg_channel)
        print("EOG:", eog_channels)
        print("EMG:", emg_channel)

        # -------------------------
        # YASA
        # -------------------------
        sls = yasa.SleepStaging(
            raw,
            eeg_name=eeg_channel,
            eog_name=eog_channels[0] if eog_channels else None,
            emg_name=emg_channel,
        )

        hypno = sls.predict()
        hypno_stages = hypno.hypno if hasattr(hypno, "hypno") else hypno
        hypno_values = hypno_stages.values if hasattr(hypno_stages, "values") else hypno_stages
        hypno_time = hypno_stages.index if hasattr(hypno_stages, "index") else range(len(hypno_values))

        # -------------------------
        # Save CSV
        # -------------------------
        df = pd.DataFrame({
            "Subject": subject,
            "Session": session,
            "Recording": recording,
            "RecordingEpoch": range(len(hypno_values)),
            "Time": hypno_time,
            "Stage": hypno_values,
        })
        df.insert(0, "Epoch", range(len(df)))

        patient_results.setdefault(subject, []).append(df.copy())

        df.to_csv(
            os.path.join(save_dir, "hypnogram.csv"),
            index=False,
        )

        stage_counts = df["Stage"].value_counts().rename_axis("Stage").reset_index(name="Count")
        stage_counts.to_csv(
            os.path.join(save_dir, "stage_counts.csv"),
            index=False,
        )

        # -------------------------
        # Save txt
        # -------------------------
        with open(
            os.path.join(save_dir, "hypnogram.txt"),
            "w",
        ) as f:
            for s in hypno_values:
                f.write(str(s) + "\n")

        # -------------------------
        # Save plot
        # -------------------------
        if hasattr(hypno, "plot_hypnogram"):
            ax = hypno.plot_hypnogram()
            fig = ax.figure
        else:
            fig = yasa.plot_hypnogram(hypno_stages)

        fig.savefig(
            os.path.join(save_dir, "hypnogram.png"),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print("Saved:", save_dir)
        print(stage_counts.to_string(index=False))

    except Exception as e:
        print("FAILED:")
        print(vhdr_file)
        print(e)


for subject, dfs in sorted(patient_results.items()):
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["Time"] = pd.to_datetime(full_df["Time"], errors="coerce")
    full_df = full_df.sort_values(
        ["Time", "Session", "Recording", "RecordingEpoch"],
        na_position="last",
    ).reset_index(drop=True)
    full_df.insert(0, "PatientEpoch", range(len(full_df)))

    subject_dir = os.path.join(OUTDIR, subject)
    os.makedirs(subject_dir, exist_ok=True)

    full_csv = os.path.join(subject_dir, "full_hypnogram.csv")
    full_txt = os.path.join(subject_dir, "full_hypnogram.txt")
    full_png = os.path.join(subject_dir, "full_hypnogram.png")
    full_counts_csv = os.path.join(subject_dir, "full_stage_counts.csv")

    full_df.to_csv(full_csv, index=False)

    with open(full_txt, "w") as f:
        for stage in full_df["Stage"]:
            f.write(str(stage) + "\n")

    full_counts = full_df["Stage"].value_counts().rename_axis("Stage").reset_index(name="Count")
    full_counts.to_csv(full_counts_csv, index=False)
    plot_full_hypnogram(full_df, full_png, f"Full hypnogram - {subject}")

    print("\nSaved full patient hypnogram:", subject_dir)
    print(full_counts.to_string(index=False))


print("\nDONE")
