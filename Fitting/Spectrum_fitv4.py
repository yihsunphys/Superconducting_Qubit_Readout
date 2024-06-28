import numpy as np
import scqubits as scq
from plotly import graph_objects as go, express as px, io as pio
from LogReader import LogHandler

Spectrum_template = pio.templates["simple_white"]
Spectrum_template.layout.update(
    dict(
        title_x = 0.1, font = dict(size = 20),
        xaxis = dict(showspikes = True, spikemode = 'across', showgrid = True),
        yaxis = dict(showgrid = True, tickformat = '.2f'),
        width=960, height=720 
    )
)

class fit_spectrum():
    def __init__(self, qubit_params, flux_params, fileDict = None, select_elems = 3):
        self.Demo = fileDict == 'Demo'
        if self.Demo:
            self.flux_list = np.linspace(*flux_params)
        else:
            self.Data = LogHandler(fileDict).output()
            xdata = self.Data[1]
            self.flux = self.I2flux(xdata*1e6, *flux_params)
            self.flux_list = np.linspace(self.flux[0], self.flux[-1], flux_params[-1])
            self.current_list = np.linspace(xdata[0],xdata[-1], flux_params[-1])*1e3
        self.index_pairs = self.elements(select_elems)
        self.gen_fluxonium(qubit_params, self.flux_list, self.index_pairs)
        self.gen_plot()
    
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
                template = Spectrum_template,
                title = f'EJ = {self.fluxonium.EJ} GHz, EC = {self.fluxonium.EC} GHz, EL = {self.fluxonium.EL} GHz',
                xaxis = dict(title = r"$\phi_{ext} / \phi_{0}$"),
                yaxis = dict(title = 'Frequency (GHz)')
            )
        )
        # matrix element layout
        self.matelem = go.Figure(
            layout = go.Layout(
                template= Spectrum_template,
                title = 'Matrix element',
                xaxis = dict(title = r"$\phi_{ext} / \phi_{0}$"),
                yaxis = dict(title = 'Matrixelement'),
            )
        )
        return self.spectrum, self.matelem
    
    def add_simulation(self, matelem_inspec = False, xunit_in_flux = False):
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
                name = f"{row}-{col}",
                line = dict(color = color),
            )
    
    def add_Data(self, data, xunit_in_flux = False, Phase = False, Normalize = False):
        # add data
        ydata, xdata, zdata, _, xname, _ = data
        if xunit_in_flux: xdata, xname = (self.flux, r"$\phi_{ext} / \phi_{0}$")
        else: xdata, xname = (xdata*1e3, 'Current (mA)')
        
        ydata *=1e-9
        zdata = np.unwrap(np.angle(zdata.T)) if Phase else np.abs(zdata.T)
        if Normalize: zdata = (zdata - np.average(zdata,0))/np.std(zdata,0)
        self.spectrum.update_xaxes(title_text = xname)
        self.matelem.update_xaxes(title_text = xname)
        self.spectrum.add_heatmap(
            x = xdata, y = ydata, z = zdata,
            # zmin = 327.6E-3, zmax = 1.052,
            colorscale = 'Viridis_r', #RdBu
            showscale = False,
            name = 'Data'
        )
        xrange = [min(xdata), max(xdata)]
        yrange = [min(ydata), max(ydata)]

        self.spectrum.update_layout(xaxis_range = xrange, yaxis_range = yrange)
    
    def plot_spectrum(self, matelem_inspec = False, xunit_in_flux = False, Phase = False, Normalize = False, show = True):
        if not self.Demo: self.add_Data(self.Data, xunit_in_flux, Phase, Normalize)
        else: xunit_in_flux = True

        self.add_simulation(matelem_inspec, xunit_in_flux)
        if show:
            self.spectrum.show(renderer = 'browser')
            self.matelem.show(renderer = 'browser')

        return self.spectrum, self.matelem

if __name__ == '__main__':
    Demo = 0
    filepath = r"C:\Users\SQC\Desktop\ZCU\ZCU_Data\Test065\2024\05\Data_0504\Test065_OneTone_flux_002.hdf5"
    filepaths = [
        r'C:\Users\SQC\Desktop\ZCU\ZCU_Data\Test065\2024\05\Data_0504\Test065_TwoTone_flux_001.hdf5',
        r'C:\Users\SQC\Desktop\ZCU\ZCU_Data\Test065\2024\05\Data_0504\Test065_TwoTone_flux_002.hdf5',
        r'C:\Users\SQC\Desktop\ZCU\ZCU_Data\Test065\2024\05\Data_0504\Test065_OneTone_flux_003.hdf5',
        r'C:\Users\SQC\Desktop\ZCU\ZCU_Data\Test065\2024\05\Data_0504\Test065_TwoTone_flux_003.hdf5',
        r'C:\Users\SQC\Desktop\ZCU\ZCU_Data\Test065\2024\05\Data_0504\Test065_TwoTone_flux_004.hdf5'
    ]
    if Demo:
        simulation = fit_spectrum(
            [9, 1, 1], # EJ, EC, EL
            [0, 1, 501], #flux list: start, end, points
            'Demo',
            select_elems = 4
        )
        (spectrum, matelem) = simulation.plot_spectrum()
    else:
        fitting2 = fit_spectrum(
            [6.95, 0.8, 1.19], # EJ, EC, EL
            [460, -1580, 501], #zero phi current (uA), half phi current(uA), points
            fileDict = filepath, #FileDict can leave blank which will automatically pop up
            # select_elems = [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4)], 
            select_elems = 4, 
        )
        fitting2.plot_spectrum(Normalize=True, show=False, matelem_inspec= False, Phase=False)
        for path in filepaths:
            fitting2.add_Data(LogHandler(path).output(), Normalize=True, Phase=False)
        fitting2.spectrum.update_layout(yaxis_range = [5.9, 6.7])
        fitting2.spectrum.update_layout(xaxis_range = [-3.54, 0.46])
        fitting2.spectrum.show()
        # fitting2.matelem.show()
    pass