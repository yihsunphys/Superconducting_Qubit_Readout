import sys
sys.path.append(r"C:\Users\SQC\Desktop\ZCU\Python\Script")

from LogReader import LogHandler
from resonator_tools import circuit

path = r"C:\Users\SQC\Labber\Data\2024\05\Data_0524\Cavit72Drive_S11_004.hdf5"
log = LogHandler(path)
x,_,y,xname,_,yname = log.output()

port1 = circuit.reflection_port(f_data = x, z_data_raw = y[0])
port1.GUIfit()
# port1.autofit()
print(port1.fitresults)
# port1.plotall()