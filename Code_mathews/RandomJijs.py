from Local_Utils.utils import *
import numpy as np
import scipy
from Code_mathews.ObjectiveFunction import ObjectiveFunction

def main():

    RandomPath = r"C:\Users\MattC\OneDrive\Brainlab Code\Data\RandomJij"

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

    for network in networks:

        output_path = "Outputs/" + network + "/RandomJij"
        makedir("Outputs/" + network)
        makedir(output_path)
        print(output_path)

        NetworkRhoIJ = np.zeros([5,5,20])
        for subject in range(no_subject):
            NetworkRhoIJ[:, :, subject]= np.loadtxt(r"C:\Users\MattC\OneDrive\Brainlab Code\Outputs\SubjectFC/" + network + "/" + str(subject+1) + "_FC.csv")

        MeanRhoIJ = np.mean(NetworkRhoIJ, axis = 2)

        alphaStars = list()
        errors = list()

        for i in range(30):
            J = np.load(RandomPath + "/J_" + str(i) + ".npz")['arr_0']

            options = dict()
            options["disp"] = True;
            options["maxiter"] = 5;

            try:
                minimizer = scipy.optimize.minimize_scalar(ObjectiveFunction, args=(J, output_path, MeanRhoIJ), options=options)
                alphaStars.append(minimizer.x)
                errors.append(minimizer.fun)
            except:
                pass

        np.savetxt(output_path + "/alphastar.csv", alphaStars)
        np.savetxt(output_path + "/error.csv", errors)


if __name__ == "__main__":
    main()