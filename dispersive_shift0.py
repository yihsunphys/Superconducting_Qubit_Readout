import numpy as np
import scqubits as scq
import matplotlib.pyplot as plt
from plotly import graph_objects as go, express as px, io as pio

class fit_spectrum():
    def __init__(self, qubit_params, select_elems = 6, state = [0]):
        self.index_pairs = self.elements(select_elems,state)
        self.gen_fluxonium(qubit_params, self.index_pairs)
    
    def elements(self, select_elems, state):
        if isinstance(select_elems, int):
            index_pairs = [(row, col) for col in state for row in range(select_elems) if(row != col)]
        else:
            index_pairs = select_elems
        return index_pairs

    def gen_fluxonium(self, qubit_params, index_pairs):
        self.fluxonium = scq.Fluxonium(*qubit_params, flux = 0, cutoff = 50)
        self.freq_list = self.fluxonium.eigenvals(evals_count=20)
        self.matelem_list = self.fluxonium.matrixelement_table('n_operator',evals_count=20)
                
    def add_simulation(self):
        for row, col in self.index_pairs:
                data_i = self.freq_list[row] 
                data_j = self.freq_list[col] 
                f_ij[col].append(np.abs(data_i - data_j)),

                matelem_vals = np.abs(self.matelem_list[row,col])
                n_ij[col].append(matelem_vals),

if __name__ == '__main__':
        EJ = 6.01
        EC = 1.59
        EL = 0.165
        g = 0.001
        f_ij = [[] for i in range(6)]   # tran freq (GHz)     
        n_ij = [[] for i in range(6)]   # matrix elem
        simulation = fit_spectrum(
            [EJ, EC, EL], # EJ, EC, EL (GHz)
            select_elems = 15,
            state = [0,2]  # |g> and |f>
        )
        simulation.add_simulation()

        x = np.linspace(0, 12, 501) 
        second_excited_dispersive = g*1000*sum((n**2 * 2 * f / (f**2 - x**2))  for n, f in zip(n_ij[2], f_ij[2])) # MHz
        ground_state_dispersive = g*1000*sum((n**2 * 2 * f / (f**2 - x**2))  for n, f in zip(n_ij[0], f_ij[0])) 
        chi_gf = ground_state_dispersive - second_excited_dispersive
        plt.plot(x, chi_gf, label = '$\chi_{gf}$')
        plt.plot(x, second_excited_dispersive, label = '$\chi_f$')
        plt.ylim(-2, 2)
        plt.legend()
        plt.grid(True)
        plt.xlabel('Resonator frequency(GHz)')
        plt.ylabel('dispersive shift (MHz)')
        plt.title(f'EJ = {EJ}, EC = {EC}, EL = {EL},  g = {g}')
        plt.show()  # just for py file