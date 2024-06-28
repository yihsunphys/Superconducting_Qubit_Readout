import numpy as np
import Labber
import plotly.graph_objects as go

version = 1.0

def get_path(title: str = 'Select processing files'):
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        filetypes=[("Labber log files (*.hdf5)", "*.hdf5")],
        title=title)

    return path

def plot(xdata: np.array, ydata: np.array, zdata: np.array,
         xname: str = 'xname', yname: str = 'yname', zname: str = 'zname',
         Phase = False, Normalize = False, Transpose = False):
    zdata = np.unwrap(np.angle(zdata)) if Phase else np.abs(zdata)
   
    if len(ydata)>1:
        if Normalize: zdata = ((zdata.T - np.average(zdata,1))/np.std(zdata,1)).T
        if Transpose:
            xdata, ydata = ydata, xdata
            xname, yname = yname, xname
            zdata = zdata.T
        fig = contour(xdata,ydata,zdata,xname,yname,zname)
        return fig
    fig = go.Figure(
        layout=go.Layout(
            xaxis=dict(title=xname, showspikes=True, spikemode='across', tickformat='.2e'),
            yaxis=dict(title=zname, showspikes=True, spikemode='across', tickformat='.2e')),
        data = go.Scatter(x = xdata, y = zdata[0], name = f'{yname} ={ydata}', mode='lines+markers')
        )
    fig.show()
    return fig

def contour(xdata: np.array, ydata: np.array, zdata: np.array,
         xname: str = 'xname', yname: str = 'yname', zname: str = 'zname'):
    fig = go.Figure(layout=go.Layout(
        xaxis=dict(title=xname, showspikes=True, spikemode='across', tickformat='.2e'),
        yaxis=dict(title=yname, showspikes=True, spikemode='across', tickformat='.2e')))
    
    fig.add_heatmap(
            colorbar=dict(title=zname, tickformat='.2e'),
            x = xdata, y = ydata, z = zdata,
            colorscale = 'RdBu_r', reversescale = True,
            showscale = False
        )
    fig.show()
    return fig

def selectFromList(listItem: list, qText: str = '', returnIndex: bool = False):
    itemString = ''
    for i, j in enumerate(listItem):
        itemString += f"{i}: {j}\n"
    idx = int(input(qText + ':\n' + itemString + '=>'))
    if returnIndex:
        return idx
    return listItem[idx]

def get_data(file: str = None, log_ch_idx: int = None):
    filepath = file if file else get_path()
    log = Labber.LogFile(filepath)
    LogCh = log.getLogChannels()
    StepCh = log.getStepChannels()

    if log_ch_idx == None:
        namelist = [log["name"] for log in LogCh]
        log_ch_idx = 0 if len(LogCh) == 1 else selectFromList(
            namelist, 'Choose the log channel index from below', True)

    zdata = log.getData(LogCh[log_ch_idx]['name'])
    z_info = {'name': LogCh[log_ch_idx]['name'], 'unit': LogCh[log_ch_idx]['unit'], 'values': zdata} 

    vector = LogCh[0]['vector']
    if vector:
        x_info = {'name':'Frequency', 'unit':'Hz', 'values':log.getTraceXY()[0]}
    else:
        x_info = {keys: StepCh[0][keys] for keys in ['name','unit','values']}
    
    if np.shape(log.getData())[0]>1:
        y_info = {keys: StepCh[int(not vector)][keys] for keys in ['name','unit','values']}
    else:
        y_info = {'name':'', 'unit':'', 'values':np.array([0])}
    
    return x_info, y_info, z_info

class LogHandler():
    def __init__(self, file = None, log_ch_idx: int = None):
        info = get_data(file, log_ch_idx)
        keys = ['name', 'unit', 'values']
        for i, axis in enumerate(['x','y','z']):
            name, unit, data = map(info[i].get, keys)
            setattr(self, f'{axis}name',f'{name} ({unit})')
            setattr(self, f'{axis}data', data)

    def plot(self,**kwargs):
        '''
        Phase = False\n
        Normalize = False\n
        Transpose = False
        '''
        return plot(*self.output(),**kwargs)
    
    def output(self):
        return self.xdata, self.ydata, self.zdata, self.xname, self.yname, self.zname

if __name__ == '__main__':
    log = LogHandler(log_ch_idx=0)
    # plot(log.xdata,log.ydata,log.zdata,
    #     'Frequency (Hz)','Current (A)',log.zname,
    #     Normalize= True,Transpose=True)
    log.plot()
