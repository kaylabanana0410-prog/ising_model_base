from Local_Utils.core_local_temp import *
from Local_Utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy
from Local_Utils.FindTStar import FindTStar


def main():
    #where data comes from
    input_path = r'C:\Users\MattC\OneDrive\School\Research\ConnectomeData\HCP_Average\Aud'
    #where it will go
    output_path = 'Outputs'

    #temporary stuff
    file_name = r"\Jij_avg.csv"
    Output_filename = "AudTest_"

    #fetch the Jij csv file
    J = np.loadtxt(input_path + file_name, delimiter=",")
    print(input_path+file_name)

    RhoIJ = r"C:\Users\MattC\OneDrive\Brainlab Code\Outputs\SubjectFC\Aud\1_FC.csv"
    RhoIJ = np.loadtxt(RhoIJ)

    # Ising Parameters
    temperature_parameters = (0.002, 4, 50)  # Temperature parameters (initial tempeture, final temperature, number of steps)
    no_simulations = 1200  # Number of simulations after thermalization
    thermalize_time = 0.3  #
    start_time = time.time()

    temps = np.logspace(temperature_parameters[0], np.log10(temperature_parameters[1]), num=temperature_parameters[2])

    Simulated_FC, Critical_Temperature, E, M, S, H, meanSpins, timeCourses = generalized_ising(J,
                                                                                            temperature_parameters=temperature_parameters,
                                                                                            n_time_points=no_simulations,
                                                                                            thermalize_time=thermalize_time,
                                                                                            phi_variables=True,
                                                                                            return_tc=True,
                                                                                            type="digital",
                                                                                            temperature_distribution="log",
                                                                                            alpha=-1)

    final_time = time.time()
    delta_time = final_time - start_time



    to_save_results(temperature_parameters,J,E,M,H,S,Simulated_FC,Critical_Temperature,output_path + "/" + "Test" + "/")
    np.save(output_path + "/" + "Test" + "/mean_spins.npy", meanSpins)
    np.save(output_path + "/" + "Test" + "/timeCourses.npy", timeCourses)

    # Find t* for each subject

    tstar, error = FindTStar(simFCs=Simulated_FC, RhoIJ=RhoIJ, temps=temps, minTemp=50)

    simulated_FC_flattened = np.reshape(Simulated_FC, [25,-1])
    print(simulated_FC_flattened.shape)

    SimFC_interpolant = scipy.interpolate.interp1d(temps, simulated_FC_flattened, 'cubic')
    TstarFC = SimFC_interpolant(tstar)

    plt.imshow(np.reshape(TstarFC, [5,5]))
    plt.show()

if __name__ == '__main__':
    main()
