from scipy.optimize import curve_fit
import plotly.graph_objects as go
import numpy as np
from hdf5Reader  import get_VNA_Data, get_hdf5_path
import Labber

def get_ZCU_Data(fileDict = None, log_ch_idx = 0):
    if fileDict is None :fileDict = get_hdf5_path()
    log = Labber.LogFile(fileDict)
    xdata, zdata = log.getTraceXY()
    xname = 'Time (s)'
    zname = 'ZCU - Demodulated Value (a.u.)' 
    return xdata, zdata, xname, zname

def func(x,a,b,c,d,t2):
    return a*np.cos(2*np.pi*b*(x-c))*np.exp(-(x-c)/t2) - d

xdata, zdata, xname, zname = get_ZCU_Data()

time = xdata*1e6
fitted = np.angle(zdata)
t2_guess = 1.5

guess = ([(np.max(fitted)-np.min(fitted))*0.5, 0.5/abs(time[np.argmax(fitted)]-time[np.argmin(fitted)]), time[0], fitted[0], t2_guess])
opt,cov = curve_fit(func, time, fitted, guess, maxfev = 1000000)
err = np.sqrt(abs(np.diag(cov)))

fig = go.Figure(
    layout=go.Layout(
        title="T2R",
        xaxis = dict(title = "Time (us)", showspikes = True, spikemode = 'across', range = [time[0],time[-1]]),
        yaxis = dict(title = "Phase")
    ))
fig.add_trace(go.Scatter(x = time, y = fitted))
fig.add_trace(go.Scatter(x = time, y = func(time, *opt)))
fig.update_layout(title =  f"T2R = {opt[3]:.4f} +/- {err[3]:.4f}(us), detune = {opt[1]:.4f}+/- {err[1]:.4f}MHz")
fig.show()