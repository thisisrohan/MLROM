import numpy as np
import os

dir_name = os.getcwd() + '/saved_data/data_001'

N = 16
Re = 30
dt = 0.01
dTr = 0.25
T_transient = 500
T = 25000
N_ref = 50

data = {
    'num_wavenumbers':2*N+1,
    'num_wavenumber_pairs':N,
    'Re':Re,
    'dt':dt,
    'dTr':dTr,
    'T_transient':T_transient,
    'T':T,
    'N_ref':N_ref,
}

with open(dir_name+'/sim_data.txt', 'w') as f:
    f.write(str(data))

np.savez(
    dir_name+'/sim_data',
    num_wavenumbers=2*N+1,
    num_wavenumber_pairs=N,
    Re=Re,
    dt=dt,
    dTr=dTr,
    T_transient=T_transient,
    T=T,
    N_ref=N_ref,
)
