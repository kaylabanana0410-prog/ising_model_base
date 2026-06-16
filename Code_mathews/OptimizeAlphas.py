from Local_Utils.utils import *
import numpy as np
import scipy
from Code_mathews.ObjectiveFunction import ObjectiveFunction

def main():

    no_subject = 20
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

    makedir(r"C:\Users\MattC\OneDrive\Brainlab Code Good\Outputs2")

    for network in networks:
        for subject in range(no_subject):
            print(network)
            # where data comes from
            input_path = r'C:\Users\MattC\OneDrive\School\Research\ConnectomeData\HCP_Average/' + network
            # where it will go
            output_path = "Outputs2/" + network + "/Subject" + str(subject)

            makedir("Outputs2/" + network)
            makedir(output_path)

            # temporary stuff
            file_name = r"\Jij_avg.csv"

            # fetch the Jij csv file
            J = np.loadtxt(input_path + file_name, delimiter=",")
            print(input_path +file_name)

            RhoIJ = r"C:\Users\MattC\OneDrive\Brainlab Code Good\Outputs\SubjectFC/" + network + "/" + str(subject+1) + "_FC.csv"
            RhoIJ = np.loadtxt(RhoIJ)

            options = dict()
            options["disp"] = True;
            options["maxiter"] = 20;

            minimizer = scipy.optimize.minimize_scalar(ObjectiveFunction, args = (J, output_path, RhoIJ), options = options)

            print("alpha is ", minimizer.x)
            print("The error is ", minimizer.fun)

            np.savetxt(output_path + "/alphastar.csv", [minimizer.x, 1])
            np.savetxt(output_path + "/error.csv", [minimizer.fun, 1])

if __name__ == "__main__":
    main()