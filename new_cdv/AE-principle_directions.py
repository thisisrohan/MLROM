#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

import time as time
import platform as platform

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L2
import h5py

tf.keras.backend.set_floatx('float32')

plt.rcParams.update({
    "text.usetex":True,
    "font.family":"serif",
})


# In[2]:


colab_flag = False
FTYPE = np.float32
ITYPE = np.int32

array = np.array
float32 = np.float32
int32 = np.int32
float64 = np.float64
int64 = np.int64

strategy = None
# strategy = tf.distribute.MirroredStrategy()

xlabel_kwargs = {"fontsize":15}
ylabel_kwargs = {"fontsize":15}
legend_kwargs = {"fontsize":12}
title_kwargs = {"fontsize":18}


# In[3]:


current_sys = platform.system()

if current_sys == 'Windows':
    dir_sep = '\\'
else:
    dir_sep = '/'


# In[4]:


if colab_flag == True:
    from google.colab import drive
    drive.mount('/content/drive')
    os.chdir('/content/drive/MyDrive/Github/MLROM/KS/')


# In[5]:


print(os.getcwd())


# In[6]:


from tools.ae_v2 import Autoencoder


# In[7]:


gpus = tf.config.list_physical_devices('GPU')
print(gpus)

if colab_flag == False:
    if strategy is None:
        if gpus:
            gpu_to_use = 0
            tf.config.set_visible_devices(gpus[gpu_to_use], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    print(logical_devices)


# In[8]:


# print(tf.test.gpu_device_name())
print(tf.config.list_physical_devices())
print('')
print(tf.config.list_logical_devices())
print('')
print(tf.__version__)


# # Kolmogorov Flow System

# In[9]:


prng_seed = 42
np.random.seed(prng_seed)

train_split = 0.8
val_split = 0.1
test_split = 0.1


# In[10]:


ae_idx = '008'
dir_name_ae = os.getcwd()+'{ds}saved_ae{ds}ae_'.format(ds=dir_sep)+ae_idx
print(dir_name_ae)

# reading simulation parameters
with open(dir_name_ae + dir_sep + 'ae_data.txt') as f:
    lines = f.readlines()
params_dict = eval(''.join(lines))
data_dir_idx = params_dict['data_dir_idx']
normalizeforae_flag = params_dict['normalizeforae_flag']
ae_module = params_dict['module']
try:
    use_ae_data = params_rnn_dict['use_ae_data']
except:
    print("'use_ae_data' not present in RNN_specific_data, set to True.")
    use_ae_data = True
try:
    ae_data_with_params = params_dict['ae_data_with_params']
except:
    print("'ae_data_with_params' not present in ae_data, set to 'True'.")
    ae_data_with_params = True

if os.path.exists(dir_name_ae+dir_sep+'normalization_data.npz'):
    with np.load(dir_name_ae+dir_sep+'normalization_data.npz', allow_pickle=True) as fl:
        normalization_constant_arr_aedata = fl['normalization_constant_arr_aedata'][0]
print('data_dir_idx:', data_dir_idx)

# loading data
dir_name_data = os.getcwd() + dir_sep + 'saved_data' + dir_sep + 'data_' + data_dir_idx

with open(dir_name_data + dir_sep + 'sim_data_params.txt') as f:
    lines = f.readlines()
params_dict = eval(''.join(lines))
params_mat = params_dict['params_mat']
# init_state = params_dict['init_state']
t0 = params_dict['t0']
T = params_dict['T']
delta_t = params_dict['delta_t']
return_params_arr = params_dict['return_params_arr']
normalize_flag_ogdata = params_dict['normalize_flag']
print('normalize_flag_ogdata:', normalize_flag_ogdata)
alldata_withparams_flag = params_dict['alldata_withparams_flag']

with np.load(dir_name_data+dir_sep+'data.npz', allow_pickle=True) as fl:
    all_data_og = fl['all_data'].astype(FTYPE)
    boundary_idx_arr = fl['boundary_idx_arr']
    normalization_constant_arr_ogdata = fl['normalization_constant_arr'][0]
    # initial_t0 = fl['initial_t0']
    # init_state_mat = fl['init_state_mat']

    lyapunov_spectrum_mat = fl['lyapunov_spectrum_mat']

num_params = params_mat.shape[1]
og_vars = all_data_og.shape[1]
if alldata_withparams_flag == True:
    og_vars -= num_params


# In[11]:


if ae_data_with_params == False:
    all_data_og = all_data_og[:, 0:og_vars]
    normalization_constant_arr_aedata = normalization_constant_arr_aedata[:, 0:og_vars]


# In[12]:


print(all_data_og.shape)


# In[ ]:





# In[13]:


num_train = int(all_data_og.shape[0]*train_split)
num_val = int(all_data_og.shape[0]*val_split)
num_test = all_data_og.shape[0] - num_train - num_val

idx = np.arange(all_data_og.shape[0])
# np.random.shuffle(idx)

training_data = np.empty(shape=(num_train, ) + tuple(all_data_og.shape[1:]), dtype=FTYPE)
val_data = np.empty(shape=(num_val, ) + tuple(all_data_og.shape[1:]), dtype=FTYPE)
testing_data = np.empty(shape=(num_test, ) + tuple(all_data_og.shape[1:]), dtype=FTYPE)

training_data[:] = all_data_og[idx[0:num_train]]
val_data[:] = all_data_og[idx[num_train:num_train+num_val]]
testing_data[:] = all_data_og[idx[num_train+num_val:]]

del(all_data_og)


# In[14]:


print('training_data.shape : ', training_data.shape)
print('val_data.shape : ', val_data.shape)
print('testing_data.shape : ', testing_data.shape)


# In[15]:


if normalizeforae_flag == True:
    training_data -= normalization_constant_arr_aedata[0]
    training_data /= normalization_constant_arr_aedata[1]

    testing_data -= normalization_constant_arr_aedata[0]
    testing_data /= normalization_constant_arr_aedata[1]


# In[16]:


load_file = dir_name_ae+dir_sep+'final_net'+dir_sep+'final_net_class_dict.txt'
wt_file = dir_name_ae+dir_sep+'final_net'+dir_sep+'final_net_ae_weights.h5'

ae_net = Autoencoder(data_dim=training_data.shape[1], load_file=load_file)
ae_net.load_weights_from_file(wt_file)


# In[17]:


latent_states_all_trainingdata = np.array(ae_net.encoder_net.predict(training_data))
latent_states_all_testingdata = np.array(ae_net.encoder_net.predict(testing_data))

# reconstructed_data_trainingdata = ae_net.decoder_net.predict(latent_states_all_trainingdata)
# reconstructed_data_testingdata = ae_net.decoder_net.predict(latent_states_all_testingdata)


# In[18]:


ls_shape = latent_states_all_testingdata.shape[1:]
og_shape = training_data.shape[1:]


# In[19]:


print('ls_shape : ', ls_shape)
print('og_shape : ', og_shape)


# In[20]:


print('latent_states_all_trainingdata.shape : {}'.format(latent_states_all_trainingdata.shape))
print('latent_states_all_testingdata.shape : {}\n'.format(latent_states_all_testingdata.shape))

# print('reconstructed_data_trainingdata.shape : {}'.format(reconstructed_data_trainingdata.shape))
# print('reconstructed_data_testingdata.shape : {}'.format(reconstructed_data_testingdata.shape))


# In[21]:


if normalizeforae_flag == True:
    # reconstructed_data_trainingdata *= normalization_constant_arr_aedata[1]
    # reconstructed_data_trainingdata += normalization_constant_arr_aedata[0]
    training_data *= normalization_constant_arr_aedata[1]
    training_data += normalization_constant_arr_aedata[0]

    # reconstructed_data_testingdata *= normalization_constant_arr_aedata[1]
    # reconstructed_data_testingdata += normalization_constant_arr_aedata[0]
    testing_data *= normalization_constant_arr_aedata[1]
    testing_data += normalization_constant_arr_aedata[0]


# In[22]:


# POD in latent space
latent_states_all_trainingdata = np.reshape(latent_states_all_trainingdata, (latent_states_all_trainingdata.shape[0], -1))
mean_ls = np.mean(latent_states_all_trainingdata, axis=0)
meancentered_ls = latent_states_all_trainingdata - mean_ls
covmat_ls = np.matmul(meancentered_ls.transpose(), meancentered_ls) / (meancentered_ls.shape[0] - 1)

eigenvals_ls, eigenvecs_ls = linalg.eig(covmat_ls)
eigenvals_ls = np.abs(eigenvals_ls)
sorted_idx = np.argsort(eigenvals_ls)
eigenvals_ls = eigenvals_ls[sorted_idx]
eigenvecs_ls = eigenvecs_ls[:, sorted_idx]

del(covmat_ls)


# In[23]:


eigenvecs_ls.shape


# In[24]:


cum_eigs = np.cumsum(eigenvals_ls[::-1])/np.sum(eigenvals_ls)

plt.semilogy(
    np.arange(1, cum_eigs.shape[0]+1),
    cum_eigs, marker='^', linestyle='--', linewidth=0.8)
plt.grid(which='both')
plt.ylim(None,1.05)

plt.xticks(np.arange(1, cum_eigs.shape[0]+1))

plt.xlabel('Index $i$', **xlabel_kwargs)
# plt.ylabel(r'$$1 - \left( \frac{\sum_{k=1}^i \lambda_k}{\sum_{k} \lambda_k} \right)$$', **ylabel_kwargs)
plt.ylabel(r'$$\frac{\sum_{k=1}^i \lambda_k}{\sum_{k} \lambda_k}$$', **ylabel_kwargs)
plt.title('Cumulative Variance of the \nLatent Space Principal Directions', **title_kwargs)

plt.gcf().set_size_inches(4, 3)

plt.savefig(dir_name_ae+'/plots/ls_cumvar.pdf', dpi=300, bbox_inches='tight')
# plt.show()
plt.close()


# In[25]:


# POD in actual-data space
training_data = np.reshape(training_data, (training_data.shape[0], -1))
mean_og = np.mean(training_data, axis=0)
meancentered_og = training_data - mean_og
covmat_og = np.matmul(meancentered_og.transpose(), meancentered_og) / (meancentered_og.shape[0] - 1)

eigenvals_og, eigenvecs_og = linalg.eig(covmat_og)
eigenvals_og = np.abs(eigenvals_og)
sorted_idx = np.argsort(eigenvals_og)
eigenvals_og = eigenvals_og[sorted_idx]
eigenvecs_og = eigenvecs_og[:, sorted_idx]

del(covmat_og)


# In[26]:


og_eigvecs_to_analyse = 6
ls_eigvecs_to_analyse = 5

eigenvecs_og_norm = linalg.norm(eigenvecs_og[:, -og_eigvecs_to_analyse:].transpose(), axis=1)

correlation_mat = np.empty(shape=(ls_eigvecs_to_analyse, og_eigvecs_to_analyse))

ls_trainingdata = latent_states_all_trainingdata - mean_ls
ls_trainingdata_norm = linalg.norm(ls_trainingdata, axis=1)

eigvecs_ls_norm = linalg.norm(eigenvecs_ls.transpose(), axis=1)

dummy_ones_mat = np.ones_like(latent_states_all_trainingdata, dtype=FTYPE)


# In[27]:


print('eigenvecs_og_norm : ', eigenvecs_og_norm)
print('eigvecs_ls_norm : ', eigvecs_ls_norm)


# In[28]:


# ls_eigenval_sum = 1.0
ls_eigenval_sum = np.sum(eigenvals_ls)
for i in range(ls_eigvecs_to_analyse):
    time_taken = time.time()
    # temp_ls = (eigenvals_ls[-i-1]/ls_eigenval_sum) * eigenvecs_ls[:, -i-1]
    # temp_ls = eigenvecs_ls[:, -i-1]
    temp_ls = (eigenvals_ls[-i-1]/ls_eigenval_sum**0.5) * eigenvecs_ls[:, -i-1]
    temp_ls += mean_ls
    temp_ls = np.reshape(temp_ls, (1,)+tuple(ls_shape))
    
    decoded_ls = np.array(ae_net.decoder_net.predict(temp_ls))
    
    decoded_ls *= normalization_constant_arr_aedata[1]
    decoded_ls += normalization_constant_arr_aedata[0]
    
    decoded_ls = np.reshape(decoded_ls, (decoded_ls.shape[0], -1))
    decoded_ls -= mean_og
    decoded_ls_norm = np.linalg.norm(decoded_ls, axis=1)

    for j in range(og_eigvecs_to_analyse):
        time_taken_j = time.time()
        coeffs = np.sum(decoded_ls * eigenvecs_og[:, -j-1], axis=1)
        coeffs /= decoded_ls_norm * eigenvecs_og_norm[-j-1]
        correlation_mat[i, j] = np.mean(coeffs)
        time_taken_j = time.time() - time_taken_j
#         print('    time_taken_j : {:02d}h {:02d}m {:02d}s'.format(
#             int(time_taken_j // 3600),
#             int((time_taken_j//60)%60),
#             int(time_taken_j % 60)
#         ))
        
    del(decoded_ls)
    del(temp_ls)
    
    time_taken = time.time() - time_taken
#     print('time_taken : {:02d}h {:02d}m {:02d}s\n'.format(
#         int(time_taken // 3600),
#         int((time_taken//60)%60),
#         int(time_taken % 60)
#     ))


# In[32]:


fig, ax = plt.subplots()
im = ax.imshow(correlation_mat, vmin=-1.0, vmax=1.0, cmap='bwr')
# a = plt.gca().get_xticks()
# ls_eigvecs_to_analyse = 16

ax.set_xticks(
    np.arange(0, og_eigvecs_to_analyse, 1),
    labels=np.arange(1, og_eigvecs_to_analyse+1, 1),
    **legend_kwargs,
)
ax.set_yticks(
    np.arange(0, ls_eigvecs_to_analyse, 1),
    labels=np.arange(1, ls_eigvecs_to_analyse+1, 1),
    **legend_kwargs,
)

ax.set_title(
    'Correlation of Decoded Latent\nSpace Principal Directions w.r.t.\nthose of the Actual Data',
    **title_kwargs,
)

ax.set_xlabel('Index $j$', **xlabel_kwargs)
ax.set_ylabel('Index $i$', **ylabel_kwargs)

fig.subplots_adjust(
    bottom=0.2,
    # left=0.1,
    # top=1.0-0.19,
)

# original data and recon data colorbar
cb_xbegin = ax.transData.transform([-0.5 + 0.5, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax.transData.transform([correlation_mat.shape[-1]-0.5 - 0.5, 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]

cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar = fig.colorbar(im, cax=cb_ax, orientation='horizontal')
# cbar = plt.colorbar(im)
cbar.set_label(
    r'$\frac{\mathcal{D} \left( \mathbf{p}^{ls}_i \right) \cdot \mathbf{p}^{true}_j}{\|\mathcal{D} \left( \mathbf{p}^{ls}_i \right) \| \ \| \mathbf{p}^{true}_j \|}$',
    **xlabel_kwargs,
)

plt.gcf().set_size_inches(4, 3)

plt.savefig(dir_name_ae+'/plots/ls_vs_actual_principaldirections.pdf', dpi=300, bbox_inches='tight')
# plt.show()
plt.close()


# In[ ]:


decoded_ls_pd_list = []

ls_eigenval_sum = 1.0
ls_eigenval_sum = np.sum(eigenvals_ls)
for i in range(ls_eigvecs_to_analyse):
    time_taken = time.time()
#     temp_ls = (eigenvals_ls[-i-1]/ls_eigenval_sum) * eigenvecs_ls[:, -i-1]
#     temp_ls = eigenvecs_ls[:, -i-1]
    temp_ls = (eigenvals_ls[-i-1]/ls_eigenval_sum**0.5) * eigenvecs_ls[:, -i-1]
    temp_ls += mean_ls
    temp_ls = np.reshape(temp_ls, (1,)+tuple(ls_shape))
    
    decoded_ls = np.array(ae_net.decoder_net.predict(temp_ls))
    
    decoded_ls *= normalization_constant_arr_aedata[1]
    decoded_ls += normalization_constant_arr_aedata[0]
    
    decoded_ls = np.reshape(decoded_ls, (decoded_ls.shape[0], -1))
    decoded_ls -= mean_og
    decoded_ls_norm = np.linalg.norm(decoded_ls, axis=1)

    decoded_ls_pd_list.append(decoded_ls)
    # del(decoded_ls)
    del(temp_ls)
    
    time_taken = time.time() - time_taken
    # print('time_taken : {:02d}h {:02d}m {:02d}s\n'.format(
    #     int(time_taken // 3600),
    #     int((time_taken//60)%60),
    #     int(time_taken % 60)
    # ))


# In[ ]:


actual_pd_list = []
for i in range(ls_eigvecs_to_analyse):
    # actual_pd_list.append(eigenvals_og[-i-1]*eigenvecs_og[:, -i-1])
    actual_pd_list.append(eigenvecs_og[:, -i-1])


# In[ ]:


np.array(decoded_ls_pd_list).shape, np.array(actual_pd_list).shape


# In[ ]:


ls_eigvecs_to_analyse = 5

fig, ax = plt.subplots(
    int(np.round(ls_eigvecs_to_analyse/2 + 0.5)), 2,
    figsize=np.array([6*2, 3.0*ls_eigvecs_to_analyse])*0.4
)
# fig, ax = plt.subplots()
if len(ax.shape) == 1:
    ax = np.array([[elem] for elem in ax])

xtick_pos = np.linspace(0, decoded_ls_pd_list[0].shape[-1]-1, 6, dtype=np.int32)
xtick_labels = xtick_pos

for i in range(ax.shape[0]*ax.shape[1]):
    plotidx1 = i//2
    plotidx2 = i%2
    print(plotidx1,plotidx2)
    if i < ls_eigvecs_to_analyse:
        vecls = decoded_ls_pd_list[i][0]
        vecog = actual_pd_list[i]
        vecls_norm = linalg.norm(vecls)
        vecog_norm = linalg.norm(vecog)
        vecls /= vecls_norm
        vecog /= vecog_norm

        kwargs = [{}, {}]
        if i == 0:
            kwargs[0]['label'] = r'$\mathcal{D} \left( \mathbf{p}_i^{ls} \right)$'
            kwargs[1]['label'] = r'$\mathbf{p}_i^{true}$'
        ax[plotidx1, plotidx2].plot(vecls, **kwargs[0])
        ax[plotidx1, plotidx2].plot(vecog, **kwargs[1])
        # ax[plotidx1, plotidx2].legend(**legend_kwargs)

        ax[plotidx1, plotidx2].grid(True)
        ax[plotidx1, plotidx2].set_xticks(
            xtick_pos,
            np.int32(xtick_labels),
            #rotation=270+45,
            **legend_kwargs
        )
        # ax[plotidx1, plotidx2].set_aspect('equal')
        # ax[plotidx1, plotidx2].set_xlabel(r'$x$', **xlabel_kwargs)
        ax[plotidx1, plotidx2].set_title(r'Index $'+str(i+1)+'$', **title_kwargs)
        
        ax[plotidx1, plotidx2].set_xlabel(r'Index $j$', **xlabel_kwargs)
        ax[plotidx1, plotidx2].set_xlabel(r'$x_j$', **ylabel_kwargs)
        
    else:
        ax[plotidx1, plotidx2].set_visible(False)

fig.legend(loc=9, **legend_kwargs)
        
plt.tight_layout()

plt.savefig(dir_name_ae+'/plots/ls_and_actual_pd.pdf', dpi=300, bbox_inches='tight')
# plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


ls_eigvecs_to_analyse = 6
fig, ax = plt.subplots(
    ls_eigvecs_to_analyse, 2,
    figsize=(5.0*2, 5.0*ls_eigvecs_to_analyse)
)
# fig, ax = plt.subplots()

actual_pd_list = np.array(actual_pd_list)

xtick_pos = np.linspace(0, vec.shape[-1]-1, 5, dtype=np.int32)
xtick_labels = [r'0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
for i in range(ls_eigvecs_to_analyse):
    vec = actual_pd_list[i]
    vec = np.reshape(vec, og_shape)
    
    u_im = ax[i, 0].imshow(
        vec[0].transpose(),
        origin='lower',
        aspect='equal',
        # vmin=vmin0,
        # vmax=vmax0,
    )
    v_im = ax[i, 1].imshow(
        vec[1].transpose(),
        origin='lower',
        aspect='equal',
        # vmin=vmin1,
        # vmax=vmax1,
    )
    
    
    
    plt.colorbar(u_im, ax=ax[i, 0], orientation='horizontal')
    plt.colorbar(v_im, ax=ax[i, 1], orientation='horizontal')
    
    for j in range(2):
        ax[i, j].set_xticks(xtick_pos, xtick_labels, **legend_kwargs)
        ax[i, j].set_xlabel(r'$x$', **xlabel_kwargs)
        ax[i, j].set_yticks(xtick_pos, xtick_labels, **legend_kwargs)
        ax[i, j].set_ylabel(r'$y$', **xlabel_kwargs)
        
    ax[i, 0].set_title(r"$u$", **title_kwargs)
    ax[i, 1].set_title(r"$v$", **title_kwargs)

plt.tight_layout()
# plt.show()


# In[ ]:




