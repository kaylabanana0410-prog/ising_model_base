import numpy as np
import scipy.stats as sp
import scipy.integrate as int
import matplotlib.pyplot as plt
import ising as I
import utils
import config as cf
import os
import pickle


class simulated_FC_vs_T_global:

    def __init__(self, min_temp, max_temp, temp_step, alpha,
                 Jij = cf.avg_Jij, ising = I.Jij_sorted_ising,
                 multiplier = utils.normalize_array(cf.ind_avg_Jij),
                 save = False):

        '''
        Runs multiple Ising model simulations at increasing temperatures, then graphs temperature vs average energy,
        average magnetization, specific heat, and magnetic susceptibility at the end.

        :param min_temp: starting temperature value
        :param max_temp: ending temperature value
        :param temp_step: number of temperature steps to get from start to end
        :param alpha: alpha value
        :param Jij: Jij matrix used for all simulations
        :param ising: Ising timescale used for all simulations
        :param multiplier: temperature multiplier values used per neuron
        :param save: set to True if you want to save data under simulation data/temp sweep data
        '''

        self.T_global = np.linspace(min_temp, max_temp, temp_step)
        self.alpha = alpha
        self.multiplier = multiplier
        self.Jij = Jij
        self.ising = ising

        self.ising_ar = []
        self.suscept_ar = []
        self.spec_heat_ar = []
        self.avg_temp_ar = []
        self.Jij_auc_ar, self.FC_auc_ar = [], []
        self.corr_ar_1, self.corr_ar_2, self.corr_ar_3, self.corr_ar_total = [], [], [], []

        self.save = save

    def simulate(self, steps, thermalization = None, spin_array = np.random.choice([-1, 1], 84),
                 partial = True, show = False, diag = False, text = True,
                 name = 'temp_sweep', path = 'simulation data/temp sweep data/'):

        '''
        Main class for preforming the temperature sweep simulations

        :param steps: number of timesteps per simulation
        :param thermalization: number of thermalization steps per simulation
        :param spin_array: set initial spin configuration for all simulations. Default is randomized
        :param partial: if True, uses partial correlation to calculate FC matrix
        :param show: if True, displays a live plot of all parameters as simulation runs
        :param diag: if True, includes diagonal values in correlation calculation between emp. and sim. FC
        :param text: if True, prints out text for each simulation that displays the final parameter values
        :param name: set custom file name
        :param path: set custom save path
        :return:
        '''

        def save():
            '''
            If self.save == True, this function will generate a log containing information about the simulations
            '''

            self.message = f'SIMULATION LOG\n' \
                           f'alpha: {self.alpha}\n' \
                           f'temp range: {self.T_global[0]}-{self.T_global[-1]}\n' \
                           f'temp steps: {np.size(self.T_global)}\n' \
                           f'critical temperature: {self.crit_temp}\n' \
                           f'mean critical temperature: {np.mean((self.multiplier ** self.alpha) * self.crit_temp)}\n' \
                           f'best temperature: {self.best_temp}\n' \
                           f'mean best temperature: {np.mean((self.multiplier ** self.alpha) * self.best_temp)}\n' \
                           f'partial correlation: {partial}\n' \
                           f'include diagonals: {diag}\n' \
                           f'time scale: {self.ising}\n' \
                           '----------------------------------\n' \
                           'highest correlation run:\n' \
                           f'{self.best_ising} \n' \
                           '----------------------------------\n' \
                           'critical temperature run:\n' \
                           f'{self.crit_ising} \n'
            import os

            # make sure the folder exists
            if not os.path.exists(path):
                os.makedirs(path)

            # count existing run folders
            num_folders = len([
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
            ])

            self.folder_name = name + '_run_' + str(num_folders)
            self.path = path + self.folder_name
            os.mkdir(self.path)

            with open(self.path + '/log.txt', 'w') as dir:
                dir.write(self.message)

            pickle.dump(self.best_ising, open(self.path + '/best_ising.pickle', 'wb'))
            pickle.dump(self.crit_ising, open(self.path + '/crit_ising.pickle', 'wb'))

        if partial:
            emp_FC1 = cf.FC_1p
            emp_FC2 = cf.FC_2p
            emp_FC3 = cf.FC_3p
            avg_FC = cf.avg_FCp
        else:
            emp_FC1 = cf.FC_1
            emp_FC2 = cf.FC_2
            emp_FC3 = cf.FC_3
            avg_FC = cf.avg_FC

        if show:
            plt.ion()
        for temp in self.T_global:
            temp_ar = temp * (self.multiplier ** self.alpha)
            avg_temp = np.mean(temp_ar)
            beta = 1 / temp
            ising = self.ising(temp_ar, Jij=self.Jij, spin_ar=spin_array)
            ising.simulate(steps, thermalization)
            sim_FC = ising.generate_FC(partial)
            ising_data = I.get_data(ising, beta, temp, self.alpha, emp_FC=avg_FC, diag=diag)
            if text:
                print(ising_data)
                print('_____________________________')

            Jij_tpr_ar, Jij_fpr_ar, Jij_auc = utils.receiver_operating_characteristic(sim_FC, self.Jij)
            FC_tpr_ar, FC_fpr_ar, FC_auc = utils.receiver_operating_characteristic(sim_FC, avg_FC)
            self.Jij_auc_ar.append(Jij_auc)
            self.FC_auc_ar.append(FC_auc)

            self.avg_temp_ar.append(avg_temp)
            self.corr_ar_1.append(ising.correlation(emp_FC1, diag))
            self.corr_ar_2.append(ising.correlation(emp_FC2, diag))
            self.corr_ar_3.append(ising.correlation(emp_FC3, diag))
            self.corr_ar_total.append(ising.correlation(avg_FC, diag))
            self.ising_ar.append(ising_data)
            self.suscept_ar.append(ising.susceptibility(beta))
            self.spec_heat_ar.append(ising.specific_heat(beta))

            if show:
                if temp != self.T_global[0]:
                    plt.close()
                figure, axis = ising_data.graph_everything(show=False)
                figure.canvas.draw()
                figure.canvas.flush_events()
        if show:
            plt.ioff()

        crit_index = np.nanargmax(self.spec_heat_ar)
        self.crit_temp = self.T_global[crit_index]
        self.crit_corr = self.corr_ar_total[crit_index]
        self.crit_ising = self.ising_ar[crit_index]
        best_corr_index = np.nanargmax(self.corr_ar_total)
        self.best_temp = self.T_global[best_corr_index]
        self.best_corr = self.corr_ar_total[best_corr_index]
        self.best_ising = self.ising_ar[best_corr_index]
        if text:
            print('critical temperature:', self.crit_temp)
            print('highest correlation run:')
            print(self.best_ising)

        if self.save:
            save()

    def graph_data(self, show = True):
        '''
        graphs temperature vs magnetic susceptability, specific heat, and correlation between sim and emp FC. If
        self.save == True, graph is saved as a PNG file as well

        :param show: if True, displays the graph at the end of temperature sweep
        :return: graph figure and axis objects
        '''

        figure, axis = plt.subplots(1, 3)
        figure.set_size_inches(15, 5)
        figure.tight_layout(pad = 3)
        for i in range(3):
            axis[i].set_box_aspect(1)
        axis[0].plot(self.T_global, self.suscept_ar)
        axis[0].set_xlabel('global temperature')
        axis[0].set_ylabel('suscpetability')
        axis[1].plot(self.T_global, self.spec_heat_ar, label='crit temp: {:.2f}'.format(self.crit_temp))
        axis[1].set_xlabel('global temperature')
        axis[1].set_ylabel('specific heat')
        axis[2].plot(self.T_global, self.corr_ar_total, 'r', label='average')
        axis[2].plot(self.T_global, self.corr_ar_1, 'g', label='FC1')
        axis[2].plot(self.T_global, self.corr_ar_2, 'b', label='FC2')
        axis[2].plot(self.T_global, self.corr_ar_3, 'y', label='FC3')
        axis[2].set_xlabel('global temperature')
        axis[2].set_ylabel('correlation')
        axis[2].legend()

        if self.save:
            figure.savefig(self.path + '/summary_graph.png')

        if show:
            plt.show()
        else:
            return figure, axis

    def graph_auc(self, show = True):
        '''
        graphs temperature vs area under curve between sim and emp FC. If
        self.save == True, graph is saved as a PNG file as well

        :param show: if True, displays the graph at the end of temperature sweep
        :return: graph figure and axis objects
        '''

        figure, axis = plt.subplots(1)
        axis.plot(self.T_global, self.Jij_auc_ar, label='Jij')
        axis.set_xlabel('global temperature')
        axis.set_ylabel('auc')
        axis.plot(self.T_global, self.FC_auc_ar, label='emp FC')
        axis.legend()

        if self.save:
            figure.savefig(self.path + '/auc_graph.png')

        if show:
            plt.show()
        else:
            return figure, axis

    def graph_matrix(self, show = True):
        '''
        Shows the Jij, sim FC, and emp FC for the temperature that had the highest correlation between sim and emp FC.
        If self.save == True, graph is saved as a PNG file as well

        :param show: if True, displays the graph at the end of temperature sweep
        :return: graph figure and axis objects
        '''

        figure, axis = self.best_ising.graph_FC(False)

        if self.save:
            figure.savefig(self.path + '/best_FC_graph.png')

        if show:
            plt.show()
        else:
            return figure, axis



if __name__ == '__main__':
    steps = 2000
    thermalization = 1000
    min_temp = 2
    max_temp = 10
    temp_step = 50
    alpha = 2.07
    simulation = simulated_FC_vs_T_global(min_temp, max_temp, temp_step, alpha)
    simulation.simulate(steps, thermalization)