import numpy as np
import os
import string
from Local_Utils.phi_utils import load_matrix
from Local_Utils.utils import makedir
import matplotlib.pyplot as plt

time_series_Path = r"E:\Soddu Preprocessing\preprocessing"
Output_Path = r"C:\Users\MattC\OneDrive\Brainlab Code\Outputs\SubjectFC"
no_subject = 20

min_temps = [53,52,52,65,56,47,57,93,59,43,84]

networks = [
    "Aud",
    "CO",
    "CP",
    "DMN",
    "Dorsal",
    "FP",
    "RS",
    "SMHand",
    "SMMouth",
    "Ventral",
    "Visual"
]

def extract_rho(path):

    time_series = np.squeeze(load_matrix(path))

    shape_ts = time_series.shape

    assert len(shape_ts) == 2
    assert shape_ts[0] != shape_ts[1]

    if shape_ts[0]>shape_ts[1]:
        time_series=time_series.T


    rho = np.corrcoef(time_series)

    return rho

time_series_paths = list()

for root, dir, file in os.walk(time_series_Path):

    filename = str(file).strip(string.punctuation)

    if "time" in filename:
        filePath = os.path.join(root, filename)

        time_series_paths.append(filePath)

i = 0
for subject in range(no_subject):
    for network in networks:
        makedir(Output_Path + "/" + network)
        print(time_series_paths[i])
        np.savetxt(Output_Path + "/" + network + "/" + str(subject+1) + "_FC.csv", extract_rho(time_series_paths[i]))
        i += 1