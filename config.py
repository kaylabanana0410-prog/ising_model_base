import utils
import numpy as np
import os

'''
Loads all data that may be used in simulations

p: FC matrix constructed from partial correlation
b: FC matrix that has been binarized
'''

avg_Jij = utils.get_matrix('Jij data_processed/avg_Jij_no_outliers_norm') # get normalized average Jij matrix with no outlier values
regions = np.shape(avg_Jij)[0] # get number of neurons


# get functional connectivity matrices
# p: FC matrix constructed from partial correlation
# b: FC matrix that has been binarized
FC_1p, FC_2p, FC_3p = utils.get_matrix('FC data_processed/avg_TS_1p'), \
                      utils.get_matrix('FC data_processed/avg_TS_2p'), \
                      utils.get_matrix('FC data_processed/avg_TS_3p')
avg_FCp = utils.average_matrices(FC_1p, FC_2p, FC_3p)

FC_1, FC_2, FC_3 = utils.get_matrix('FC data_processed/avg_TS_1'), \
                   utils.get_matrix('FC data_processed/avg_TS_2'), \
                   utils.get_matrix('FC data_processed/avg_TS_3')
avg_FC = utils.average_matrices(FC_1, FC_2, FC_3)

FC_1pb, FC_2pb, FC_3pb = utils.get_matrix('FC data_processed/avg_TS_1pb'), \
                         utils.get_matrix('FC data_processed/avg_TS_2pb'), \
                         utils.get_matrix('FC data_processed/avg_TS_3pb')
avg_FCpb = utils.average_matrices(FC_1pb, FC_2pb, FC_3pb)

# arrays that contain the average number of functional and synaptic connections per neuron
ind_avg_Jij = np.mean(avg_Jij, 0)
norm_ind_avg_Jij = utils.normalize_array(ind_avg_Jij)
ind_avg_FC = np.mean(avg_FC, 0)

# same as above but now sorted from most to least connections
sort_ind_avg_FC = ind_avg_FC.copy()
sort_ind_avg_FC.sort()
sort_ind_avg_Jij = ind_avg_Jij.copy()
sort_ind_avg_Jij.sort()
