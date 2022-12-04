import numpy as np
import os

### define grid search params
res_size = [2000, 5000]
omega_in = [0.6, 0.8, 1.]
rho_res = [0.4, 0.6, 0.8]
alpha = [0.75, 0.1]
deg_of_connectivity = [3, 20, 80, 200]
num_workers = 2 # number of gpus/threads/workers that will be used in training

### saving grid search params
dir_gs = os.getcwd() + '/grid_search'
if not os.path.isdir(dir_gs):
    os.makedirs(dir_gs)

counter = 0
while True:
    dir_check = '/gridsearch_' + str(counter).zfill(3)
    if os.path.isdir(dir_gs + dir_check):
        counter += 1
    else:
        break

dir_gs = dir_gs + dir_check
os.makedirs(dir_gs)

np.savez(
    dir_gs+'/gridsearch_params',
    res_size=res_size,
    omega_in=omega_in,
    rho_res=rho_res,
    alpha=alpha,
    deg_of_connectivity=deg_of_connectivity,
)


total_num_params = len(res_size)*len(omega_in)*len(rho_res)*len(alpha)*len(deg_of_connectivity)

partitions = np.int32(np.linspace(0, total_num_params, num_workers+1))
big_mat = np.empty(shape=(total_num_params, 5))
counter = 0
# do not change the order below, it ensures the res_sizes get split evenly amongst the workers
for i1 in range(len(alpha)):
    d4 = alpha[i1]
    for i2 in range(len(omega_in)):
        d2 = omega_in[i2]
        for i3 in range(len(rho_res)):
            d3 = rho_res[i3]
            for i4 in range(len(deg_of_connectivity)):
                d5 = deg_of_connectivity[i4]
                for i5 in range(len(res_size)):
                    d1 = res_size[i5]
                    big_mat[counter, :] = np.array([d1, d2, d3, d4, d5])
                    counter += 1

for i in range(num_workers):
    np.savez(
        dir_gs+'/gsp_worker_{}'.format(i),
        gsp=big_mat[partitions[i]:partitions[i+1], :],
        column_names=['res_size', 'omega_in', 'rho_res', 'alpha', 'deg_of_connectivity']
    )

    
