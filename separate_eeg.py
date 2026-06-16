import mne
import yasa

raw = mne.io.read_raw_brainvision("file.vhdr", preload=True)
hypno = yasa.SleepStaging(raw).predict()
