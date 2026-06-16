import mne
import yasa
import pandas as pd

# Load BrainVision EEG
raw = mne.io.read_raw_brainvision(
    "sub-0002_task-sleep_20160531_2257.vhdr",
    preload=True
)

# Show channel names
print("Channels:")
print(raw.ch_names)

# Pick an EEG channel that exists in your data
# Change "Cz" if necessary
eeg_channel = "Cz"

# Run automatic sleep staging
sls = yasa.SleepStaging(
    raw,
    eeg_name=eeg_channel
)

# Predict stages
hypno = sls.predict()

# Save results
df = pd.DataFrame({
    "epoch": range(len(hypno)),
    "stage": hypno
})

df.to_csv("sleep_stages.csv", index=False)

print(df.head())
print("\nSaved to sleep_stages.csv")