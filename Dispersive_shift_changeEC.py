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
        self.fluxonium = scq.Fluxonium(*qubit_params, flux = 0, cutoff = 100) # flux = 0
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
    EJ = 6.0
    # EC = 1.59
    # EL = 0.165
    EL = 0.15
    g = 0.04  # 40MHz

    # Generate an array of EC values from 1 to 3
    EC_values = np.linspace(1, 3, 100)

    chi_gf_values = []
    chi_ge_values = []
    second_excited_dispersive_values = []
    
    for EC in EC_values:
        simulation = fit_spectrum(
            [EJ, EC, EL],  # EJ, EC, EL (GHz)
            select_elems=15,
            state=[0, 1, 2]  # |g> and |f>
        )
        f_ij = [[] for _ in range(6)]  # tran freq (GHz)
        n_ij = [[] for _ in range(6)]  # matrix elem
        simulation.add_simulation()
        second_excited_dispersive = g * 1000 * sum(
            (n ** 2 * 2 * f / (f ** 2 - 7.2 ** 2)) for n, f in zip(n_ij[2], f_ij[2]))  # MHz
        first_excited_dispersive = g * 1000 * sum(
            (n ** 2 * 2 * f / (f ** 2 - 7.2 ** 2)) for n, f in zip(n_ij[1], f_ij[1]))  # MHz
        ground_state_dispersive = g * 1000 * sum(
            (n ** 2 * 2 * f / (f ** 2 - 7.2 ** 2)) for n, f in zip(n_ij[0], f_ij[0]))
        chi_ge = ground_state_dispersive - first_excited_dispersive
        chi_gf = ground_state_dispersive - second_excited_dispersive
        chi_gf_values.append(chi_gf)
        chi_ge_values.append(chi_ge)
        second_excited_dispersive_values.append(second_excited_dispersive)

    plt.plot(EC_values, chi_ge_values, label='$\chi_{ge}$')
    plt.plot(EC_values, chi_gf_values, label='$\chi_{gf}$')
    
    #plt.plot(EC_values, second_excited_dispersive_values, label='$\chi_f$')
    plt.ylim(-50, 50)
    plt.legend()
    plt.grid(True)
    plt.xlabel('EC(GHz)')
    plt.ylabel('dispersive shift (MHz)')
    plt.title('Reconator freq = 5.9GHz, EJ = 6.0GHz, EL = 0.15GHz, g = 40MHz')
    plt.show()  # just for py file