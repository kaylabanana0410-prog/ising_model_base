import numpy as np
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
import ising as I
import utils
import config as cf
import pandas as pd
from datetime import date
import os
import pickle


class optimize():

    def __init__(self, ising, spins = np.random.choice([-1, 1], 84), Jij = cf.avg_Jij, partial = True, multiplier = utils.normalize_array(cf.ind_avg_Jij), save = False):
        '''
        Preforms parameter annealing to find optimal global temperature and alpha values. This is done by defining an
        error function between the simulated and empirical FC matrices and sampling random alpha and temperature values.
        The values sampled are addaptively selected based on the previous error value, such that if a simulation
        produces low error, the next set of global temperature and alpha values will be close to the previous values.

        :param ising: type of timescale used
        :param spins: set initial spin values
        :param Jij: set Jij matrix
        :param partial: if True, use partial correlation to generate simulated FC matrix
        :param multiplier: temperature multiplier per neuron
        :param save: if True, saves results under simulation data/optimization data
        '''

        self.ising = ising
        self.spins = spins
        self.Jij = Jij
        self.partial = partial
        self.multiplier = multiplier
        self.FC = []
        self.T_global, self.alpha, self.error, self.correlate = [], [], [], []
        self.run_number = 0
        self.save = save

        self.cur_date = date.today()
        self.base_dir = 'simulation data/optimization data'
        os.makedirs(self.base_dir, exist_ok=True)
        self.run_index = str(len(next(os.walk(self.base_dir))[1]))

    def save_run(self):
        '''
        If self.save == True, saves a log containing the error, correlation. global temperature, and alpha values of the
        best run
        '''

        data = {
            'error': self.error,
            'correlation': self.correlate,
            'global temp': self.T_global,
            'alpha': self.alpha
        }

        self.dataframe = pd.DataFrame(data)

        run_index = int(self.run_index)
        while True:
            save_folder_name = 'parameter optimization run ' + str(run_index) + '_' + self.cur_date.strftime("%d_%m_%Y")
            self.directory = os.path.join(self.base_dir, save_folder_name)
            try:
                os.mkdir(self.directory)
                break
            except FileExistsError:
                run_index += 1
        self.run_index = str(run_index)

        log_path = self.directory + '/log'
        with open(log_path, 'w') as file:
            error = self.optim_param.fun
            T_global, alpha = self.optim_param.x
            file.write('error: {:.2f} | highest correlation: {:.2f}\n'
                       'best global temp: {} | best alpha: {} | best average temperature: {:.2f}\n'
                       'time scale: {} | partial correlation: {}\n\n'
                       .format(error, np.max(self.correlate),
                               T_global, alpha, np.mean(T_global * self.multiplier ** alpha),
                               self.ising, self.partial))
            file.write(self.dataframe.to_string())

    def anneal(self, steps, maxfun, emp_FC, therm = None, no_local_search = True, show = False):
        '''
        Preforms the annealing operation

        :param steps: number of simulations ran
        :param maxfun: sets the "group size" for an annealing run. This sets a limit for how many times a series of
                       update algorithms can be run before resetting the annealing process and selecting new random
                       variables. This prevents the algorithm from being stuck at a local error minimum, which may not
                       be the lowest error the system may achieve
        :param emp_FC: set the emprical FC matrix to be compared to
        :param therm: set number of thermalization steps
        :param no_local_search: If true, the algorithm won't preform a "sub-annealing" operation, which basically does
                                a seperate annealing search that only restricts itself around 1 set of parameters. This
                                is usually done to get super accurate ideal parameters, but for our purposes it's not
                                necessary and is generally just a waste of time
        :param show: If True, shows a live plot of energy and spin data for each annealing run
        '''

        if show:
            iterations = np.arange(steps + 1)
            plt.ion()

        def simulate_FC(params, steps, therm):
            self.run_number += 1
            T_global, alpha = params
            self.T_global.append(T_global)
            self.alpha.append(alpha)
            temp = T_global * (self.multiplier ** alpha)

            time_series = self.ising(temp, Jij = self.Jij, spin_ar = self.spins)
            time_series.simulate(steps, thermalization = therm)
            FC = time_series.generate_FC(partial = self.partial)
            correlate = time_series.correlation(emp_FC)
            error = ((1 - correlate) + np.sqrt(np.mean((FC - emp_FC) ** 2))) ** 2

            print('{}. time: {:.2f}'.format(self.run_number, time_series.timer))
            print('   error: {:.2f} | correlation: {:.2f} | temperature: {:.2f} | alpha: {:.2f} | average temp: {:.2f}'.format(error, correlate, T_global, alpha, np.mean(temp)))
            print('---------------------------------------------------------------------------------------------------')
            self.error.append(error)
            self.correlate.append(correlate)
            self.FC.append(FC)

            if show:
                plt.close()
                beta = 1 / T_global
                ts_data = I.get_data(time_series, beta, self.T_global, self.alpha)
                figure, axis = ts_data.graph_everything(show = False)
                figure.canvas.draw()
                figure.canvas.flush_events()
            return error

        self.optim_param = dual_annealing(simulate_FC, ((0.1, 10), (-3, 3)), args = (steps, therm),
                                     no_local_search = no_local_search,
                                     maxfun = maxfun)

        if self.save:
            self.save_run()

        if show:
            plt.ioff()
        return self.optim_param

    def plot_error(self, show = True):
        np_T_global = np.array(self.T_global)
        np_alpha = np.array(self.alpha)
        np_error = np.array((self.error))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('global temp')
        ax.set_ylabel('alpha')
        ax.set_zlabel('error')

        ax.scatter(np_T_global, np_alpha, np_error)
        if self.save:
            pickle.dump(fig, open(f'{self.directory}/error graph.fig.pickle', 'wb'))

        if show:
            plt.show()

    def plot_auc(self, show = True):
        FC_auc, Jij_auc = [], []

        for sim_FC in self.FC:
            _, _, auc = utils.receiver_operating_characteristic(sim_FC, cf.avg_FC)
            FC_auc.append(auc)

            _, _, auc = utils.receiver_operating_characteristic(sim_FC, cf.avg_Jij)
            Jij_auc.append(auc)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('global temp')
        ax.set_ylabel('alpha')
        ax.set_zlabel('auc')

        ax.scatter(self.T_global, self.alpha, FC_auc)
        ax.scatter(self.T_global, self.alpha, Jij_auc)

        if self.save:
            pickle.dump(fig, open(f'{self.directory}/auc graph.fig.pickle', 'wb'))

        if show:
            plt.show()


def load_3d_plots(folder_name, file_name):
    directory = 'simulation data/optimization data/' + folder_name
    plot = utils.get_pickle_file(directory, file_name)
    plt.show()


if __name__ == '__main__':
    timescale = [I.random_ising, I.Jij_sorted_ising]
    partial = [True, False]
    for _ in range(3):
        spins = np.random.choice([-1, 1], 84)
        for ts in timescale:
            optim = optimize(ts, save = True)
            optim.anneal(2000, 500, cf.avg_FC, therm=1000, show=False)
            optim.plot_error(False)
            optim.plot_auc(False)

    #folder_name = 'parameter optimization run 34_12_03_2025'
    #file_name = 'error graph.fig.pickle'
    #load_3d_plots(folder_name, file_name)
    alpha_star = result.x[3]
    print(f"alpha* = {alpha_star:.4f}")
