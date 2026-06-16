import numpy as np
import matplotlib.pyplot as plt
import scipy

temperature_parameters = (0.002, 4, 50)
temps = np.logspace(temperature_parameters[0], np.log10(temperature_parameters[1]), num=temperature_parameters[2])

subject_no = 14
network = "Aud"

path = "Outputs/" + network + "/Subject" + str(subject_no)

Simulated_FC = np.load(path + "/sim_fc.npy")
tstar = np.loadtxt(path + "/tstar.csv")[0]
RhoIJ = r"Outputs\SubjectFC/" + network + "/" + str(subject_no) + "_FC.csv"
RhoIJ = np.loadtxt(RhoIJ)

print(tstar)

simulated_FC_flattened = np.reshape(Simulated_FC, [25,-1])
print(simulated_FC_flattened.shape)
SimFC_interpolant = scipy.interpolate.interp1d(temps, simulated_FC_flattened, 'cubic')
TstarFC = SimFC_interpolant(tstar)

[fig, axs] = plt.subplots(1,2)
axs[0].imshow(np.reshape(TstarFC, [5,5]))
axs[1].imshow(RhoIJ)
plt.show()