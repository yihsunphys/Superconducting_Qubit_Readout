import numpy as np
import scqubits as scq
from plotly import graph_objects as go, express as px
from LogReader import LogHandler

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
                template= 'simple_white',
                title = f'EJ = {self.fluxonium.EJ} GHz, EC = {self.fluxonium.EC} GHz, EL = {self.fluxonium.EL} GHz',
                title_x = 0.1, font = dict(size = 20),
                xaxis = dict(title = r"$\phi_{ext} / \phi_{0}$", showspikes = True, spikemode = 'across', showgrid = True),
                yaxis = dict(title = 'Frequency (GHz)', showgrid = True, tickformat = '.2f'),
                width=960, height= 720
                )
            )
        # matrix element layout
        self.matelem = go.Figure(
            layout = go.Layout(
                template= 'simple_white',
                title = 'Matrix element',
                title_x = 0.1, font = dict(size = 20),
                xaxis = dict(title = r"$\phi_{ext} / \phi_{0}$", showspikes = True, spikemode = 'across', showgrid = True),
                yaxis = dict(title = 'Matrixelement', showspikes = True, spikemode = 'across', showgrid = True, tickformat = '.2f'),
                width=960, height=720
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
        ydata, xdata, zdata, yname, xname, zname = data
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
            colorscale = 'Viridis', #RdBu
            showscale = False,
            name = 'Data'
        )
        xrange = [min(xdata), max(xdata)]
        yrange = [min(ydata), max(ydata)]

        self.spectrum.update_layout(xaxis_range = xrange, yaxis_range = yrange)
    
    def plot_spectrum(self, matelem_inspec = False, xunit_in_flux = False, Phase = False, Normalize = False, show = True):
        if not self.Demo: self.add_Data(self.Data, xunit_in_flux, Phase, Normalize)
        self.add_simulation(matelem_inspec, xunit_in_flux)
        if show:
            self.spectrum.show(renderer = 'browser')
            # self.matelem.show(renderer = 'browser')

        return self.spectrum, self.matelem

if __name__ == '__main__':
    Demo = 0
    filepath = r"C:\Users\SQC\Desktop\ZCU\ZCU_Data\Sapphire150_3\2023\12\Data_1231\Sapphire150_TwoTone_flux_003.hdf5"
    filepaths = [
        r'C:\Users\SQC\Desktop\ZCU\ZCU_Data\Sapphire150_3\2024\01\Data_0102\Sapphire150_TwoTone_flux_012.hdf5',
        r'C:\Users\SQC\Desktop\ZCU\ZCU_Data\Sapphire150_3\2023\12\Data_1231\Sapphire150_TwoTone_flux_004.hdf5',
        # r'C:\Users\SQC\Desktop\ZCU\ZCU_Data\Sapphire150_3\2023\12\Data_1231\Sapphire150_TwoTone_flux_007.hdf5',
        r'C:\Users\SQC\Desktop\ZCU\ZCU_Data\Sapphire150_3\2024\01\Data_0101\Sapphire150_TwoTone_flux_008.hdf5',
        r'C:\Users\SQC\Desktop\ZCU\ZCU_Data\Sapphire150_3\2023\12\Data_1231\Sapphire150_TwoTone_flux_005.hdf5',
        r'C:\Users\SQC\Desktop\ZCU\ZCU_Data\Sapphire150_3\2024\01\Data_0101\Sapphire150_TwoTone_flux_009.hdf5',
    ]
    if Demo:
        simulation = fit_spectrum(
            [2, 1, 1.5], # EJ, EC, EL
            [0, 1, 501], #flux list: start, end, points
            'Demo',
            select_elems = 5
        )
        (spectrum, matelem) = simulation.plot_spectrum(matelem_inspec = False, xunit_in_flux= True)
        
    else:
        fitting2 = fit_spectrum(
            [1.46, 0.89, 0.356], # EJ, EC, EL
            [-990, -2920, 501], #zero phi current (uA), half phi current(uA), points
            fileDict = filepath, #FileDict can leave blank which will automatically pop up
            select_elems = [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3),(0,4),(1,4),(1,5),(1,6),(6,2),(7,2)], 
            # select_elems = 4, 
        )
        fitting2.plot_spectrum(Normalize=True, show=False, matelem_inspec= False, Phase=True)
        for path in filepaths:
            fitting2.add_Data(LogHandler(path).output(), Normalize=True, Phase=True)
        fitting2.spectrum.update_layout(yaxis_range = [0.1, 8])
        fitting2.spectrum.update_layout(xaxis_range = [-3, 1])
        fitting2.spectrum.show()
        fitting2.matelem.show()
        

    pass