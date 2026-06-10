from Local_Utils.utils import to_find_critical_temperature
import numpy as np
import matplotlib.pyplot as plt
import scipy


def FindTStar(simFCs, RhoIJ, temps, minTemp):

    err_list = list()
    rhoIJ = RhoIJ[np.where(~np.eye(RhoIJ.shape[0],dtype=bool))].flatten()

    for i in range(simFCs.shape[-1]):

        simFC = simFCs[:,:,i]
        simFC_nondiag = simFC[np.where(~np.eye(simFC.shape[0],dtype=bool))].flatten()
        print(simFC_nondiag.shape)
        print(rhoIJ.shape)


        error = np.corrcoef(np.nan_to_num(simFC), rhoIJ)[1, 0] + np.sqrt(np.mean((simFC_nondiag - rhoIJ)**2))
        err_list.append(error)

    tstar = to_find_critical_temperature(np.nan_to_num(np.asarray(err_list)), temps)
    error_interpolant = scipy.interpolate.interp1d(np.asarray(temps), np.asarray(err_list), 'cubic')
    error = (1 - error_interpolant(tstar))**2;

    return tstar, error



def NCC(A, B):

    num = np.dot(A,B)
    denom1 = np.dot(A,A)
    denom2 = np. dot(B,B)

    return num / np.sqrt(denom2*denom1)



