import numpy as np
import scqubits as scq
from plotly import graph_objects as go, express as px
from hdf5Reader  import get_VNA_Data, get_hdf5_path
import Labber

def get_ZCU_Data(fileDict, log_ch_idx = 0):
    if fileDict is None :fileDict = get_hdf5_path()
    log = Labber.LogFile(fileDict)
    LogCh = log.getLogChannels()
    StepCh = log.getStepChannels()

    xdata = StepCh[0]['values']
    ydata = StepCh[1]['values']
    zdata = log.getData(LogCh[0]['name']).T
    
    xname = 'Frequency (Hz)'
    yname = 'Global - Current (A)'
    zname = 'ZCU - Demodulated Value (a.u.)' 
    record = 0
    return xdata, ydata, zdata, xname, yname, zname, record

class fit_spectrum:
    def __init__(self, qubit_params, flux_params, fileDict = None, select_elems = 3):
        """
        Simulation of fluxonium spectrum for manually fitting

        Args:
        ---
            qubit_params (list, tuple): 
                [EJ, EC, EL]\n
            flux_params (list, tuple): 
                if fileDict is 'Demo' input 3 parameter [start, end, point] to generate flux list,\n
                if a file path is provided input 3 parameter [zero phi current, half phi current, points] \
                will convert current to flux list\n
            fileDict (str, optional): 
                Input 'Demo' for only simulation, leave blank will pop up selection window. Defaults to None.\n
            select_elems (int, list of tuple): 
                either maximum index of desired matrix elements, or list [(i1, i2), (i3, i4), …] \
                of index tuples for specific desired matrix elements. Defaults to 3.
        """
        
        if fileDict == 'Demo':
            self.Demo = True
            self.flux_list = np.linspace(*flux_params)
        else:
            self.Demo = False
            if fileDict is None :fileDict = get_hdf5_path()
            try:
                self.Data = get_VNA_Data(fileDict, log_ch_idx = 0)
            except:
                self.Data = get_ZCU_Data(fileDict, log_ch_idx = 0)
            xdata = self.Data[1]
            self.flux = self.I2flux(xdata*1e6, *flux_params)
            self.flux_list = np.linspace(self.flux[0], self.flux[-1], flux_params[-1])
            self.current_list = np.linspace(xdata[0],xdata[-1], flux_params[-1])

        
        self.index_pairs = self.elements(select_elems)
        self.gen_fluxonium(qubit_params, self.flux_list, self.index_pairs)

    def I2flux(self, current, *flux_params):
        phi_zero, phi_half, _= flux_params
        phi = 2*np.abs(phi_half - phi_zero)
        if phi_zero > phi_half: phi_zero -=phi 
        return (current-phi_zero)/phi

    def elements(self, select_elems):
        if isinstance(select_elems, int):
            index_pairs = [(row, col) for row in range(select_elems) for col in range(row)]
        else:
            index_pairs = select_elems
        return index_pairs
    
    def gen_fluxonium(self, qubit_params, flux_list, index_pairs):
        self.fluxonium = scq.Fluxonium(*qubit_params, flux = 0, cutoff = 30)
        self.spectrumdata = self.fluxonium.get_spectrum_vs_paramvals(
            "flux", flux_list, np.max(index_pairs)+1)
        self.matelemdata = self.fluxonium.get_matelements_vs_paramvals(
            'n_operator', "flux", flux_list, np.max(index_pairs)+1)

    def gen_plot(self):
        # spectrum layout  
        self.spectrum = go.Figure(
            layout = go.Layout(
                title = f'EJ = {self.fluxonium.EJ} GHz, EC = {self.fluxonium.EC} GHz, EL = {self.fluxonium.EL} GHz',
                title_x = 0.1,
                xaxis = dict(title = r"$\phi_{ext} / \phi_{0}$", showspikes = True, spikemode = 'across'),
                yaxis = dict(title = 'Frequency (GHz)'),
                width=900, height=750 
                )
            )
        # matrix element layout
        self.matelem = go.Figure(
            layout = go.Layout(
                title = 'Matrix element',
                title_x = 0.1,
                xaxis = dict(title = r"$\phi_{ext} / \phi_{0}$", showspikes = True, spikemode = 'across'),
                yaxis = dict(title = 'Matrixelement', showspikes = True, spikemode = 'across'),
                width=900, height=750
            )
        )
    
    def add_simulation(self, matelem_inspec, xunit_in_flux):
        #add sqc simulation data
        colors = px.colors.qualitative.Plotly
        
        for idx, (row, col) in enumerate(self.index_pairs):
            color = colors[idx%len(colors)]
            dataset_i = self.spectrumdata.energy_table.T[row]
            dataset_j = self.spectrumdata.energy_table.T[col]
            matelem_vals = np.abs(self.matelemdata.matrixelem_table[:,row,col])
            norm_matelem_vals = (matelem_vals - np.min(matelem_vals))/(np.max(matelem_vals) - np.min(matelem_vals))

            self.spectrum.add_scatter(
                x = self.flux_list if xunit_in_flux else self.current_list,
                y = np.abs(dataset_i - dataset_j), # ij transition
                name = f'{row}-{col}',
                mode = 'markers' if matelem_inspec else 'lines',
                marker = dict(opacity = norm_matelem_vals, color = color),
                line = dict(color = color),
            )
            
            self.matelem.add_scatter(
                x = self.flux_list if xunit_in_flux else self.current_list,
                y = matelem_vals,
                name = f"{row},{col}",
                line = dict(color = color),
            )
        
    def add_Data(self, xunit_in_flux, Phase):
        # add data
        ydata, xdata, zdata, yname, xname, zname, _ = self.Data
        if xunit_in_flux: xdata, xname = (self.flux, r"$\phi_{ext} / \phi_{0}$")
        ydata *=1e-9
        zdata = np.unwrap(np.angle(zdata)) if Phase else np.abs(zdata)
        self.spectrum.update_xaxes(title_text = xname)
        self.matelem.update_xaxes(title_text = xname)
        self.spectrum.add_heatmap(
            x = xdata, y = ydata, z = zdata,
            # zmin = 327.6E-3, zmax = 1.052,
            colorscale = 'RdBu_r', reversescale = True,
            showscale = False,
            name = 'Data'
        )
        xrange = [min(xdata), max(xdata)]
        yrange = [min(ydata), max(ydata)]

        self.spectrum.update_layout(xaxis_range = xrange, yaxis_range = yrange)

    def plot_spectrum(self, matelem_inspec = False, xunit_in_flux = True, Phase = False):
        self.gen_plot()
        if not self.Demo: self.add_Data(xunit_in_flux, Phase)
        self.add_simulation(matelem_inspec, xunit_in_flux)
        self.spectrum.show(renderer = 'browser')
        # self.matelem.show(renderer = 'browser')

        return self.spectrum, self.matelem
    
if __name__ == '__main__':
    Demo = 0
    filepath = r"C:\Users\SQC\Desktop\ZCU\ZCU_Data\Sapphire Test 069\2023\10\Data_1009\Sapphire_Test_069_TwoTone_flux_025.hdf5"
    if Demo:
        simulation = fit_spectrum(
            [9.31, 0.84, 1.03], # EJ, EC, EL
            [0, 1, 501], #flux list: start, end, points
            'Demo',
            select_elems = 4
        )
        (spectrum, matelem) = simulation.plot_spectrum(matelem_inspec = True)
    else:
        fitting = fit_spectrum(
            [9.43, 0.821, 1.07], # EJ, EC, EL
            [-825, 1135, 201], #zero phi current (uA), half phi current(uA), points
            # fileDict = filepath, #FileDict can leave blank which will automatically pop up
            select_elems = 5,
        )
        (spectrum, matelem) = fitting.plot_spectrum(matelem_inspec = False, xunit_in_flux= False, Phase = False)
    pass

