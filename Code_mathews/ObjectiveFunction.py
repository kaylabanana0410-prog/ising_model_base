from Local_Utils.core_local_temp import *
from Local_Utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy
from Local_Utils.FindTStar import FindTStar

def ObjectiveFunction(alpha, J, output_path, RhoIJ):

    print(alpha)
    # Ising Parameters
    temperature_parameters = (0.002, 4, 50)  # Temperature parameters (initial tempeture, final temperature, number of steps)
    no_simulations = 1000  # Number of simulations after thermalization
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
                                                                                            alpha= alpha)

    final_time = time.time()
    delta_time = final_time - start_time


    to_save_results(temperature_parameters,J,E,M,H,S,Simulated_FC,Critical_Temperature,output_path + "/")
    np.save(output_path + "/mean_spins.npy", meanSpins)
    np.save(output_path + "/timeCourses.npy", timeCourses)

    # Find t* for each subject
    tstar, error = FindTStar(simFCs=Simulated_FC, RhoIJ=RhoIJ, temps=temps, minTemp=50)
    np.savetxt(output_path + "/tstar.csv", [tstar, 1])

    return error
