import numpy as np
import matplotlib.pyplot as plt
import h5py

################################################################################
################################# READING DATA #################################
################################################################################

fln = 'Kolmogorov_Re40.0_T12500.0_DT01.h5'

with h5py.File(fln, 'r') as f:
    t_recorded_samples = np.array(f['t'])
    snapshot_idx = int(t_recorded_samples.shape[0] / 2)

    D = np.array(f['Dissipation'])

    u_ref = np.array(f['u_reference'])
    u_mean = np.sum(u_ref, axis=0) / u_ref.shape[0]
    u_snapshot = u_ref[snapshot_idx].copy()
    del(u_ref)
    v_ref = np.array(f['v_reference'])
    v_mean = np.sum(v_ref, axis=0) / v_ref.shape[0]
    v_snapshot = v_ref[snapshot_idx].copy()
    del(v_ref)
    vort_ref = np.array(f['vort_reference'])
    vort_mean = np.sum(vort_ref, axis=0) / vort_ref.shape[0]
    vort_snapshot = vort_ref[snapshot_idx].copy()
    del(vort_ref)
    
    u = np.array(f['u'])
    u_mean_org = np.mean(u, axis=0) # np.sum(u, axis=0) / u.shape[0]
    v = np.array(f['v'])
    v_mean_org = np.mean(v, axis=0) # np.sum(v, axis=0) / v.shape[0]

T = t_recorded_samples[-1]

################################################################################
######################## COMPUTING REQUIRED QUANTITIES #########################
################################################################################

### computing KE
K = np.mean((u - u_mean_org)**2 + (v - v_mean_org)**2, axis=-1) # TKE
K = 0.5*np.mean(K, axis=-1)

### median and IQR of KE and D
D_sorted = np.sort(D)
KE_sorted = np.sort(K)

D_median = D_sorted[int(0.5*D.shape[0])]
D_Q1 = D_sorted[int(0.25*D.shape[0])]
D_Q3 = D_sorted[int(0.75*D.shape[0])]
D_IQR = D_Q3 - D_Q1

KE_median = KE_sorted[int(0.5*K.shape[0])]
KE_Q1 = KE_sorted[int(0.25*K.shape[0])]
KE_Q3 = KE_sorted[int(0.75*K.shape[0])]
KE_IQR = KE_Q3 - KE_Q1

### mean and std of KE and D
D_mean = np.mean(D)
KE_mean = np.mean(K)

D_std = np.std(D)
KE_std = np.std(K)

### correcting mean and std of KE and D
# '''
cond = True
tol = 1e-3
val1 = D_mean
val1 = np.sort(D)[int(0.5*D.shape[0])]
std_multiplier = 1.0
while(cond):
    idx = np.where(np.abs(D - val1) < std_multiplier*D_std)[0]
    val2 = np.mean(D[idx])
    D_std = np.std(D[idx])
    if np.abs(1 - val2 / val1) < tol:
        cond = False
        D_mean = val2
    else:
        val1 = val2

cond = True
tol = 1e-3
val1 = KE_mean
std_multiplier = 1.0
while(cond):
    idx = np.where(np.abs(K - val1) < std_multiplier*KE_std)[0]
    val2 = np.mean(K[idx])
    KE_std = np.std(K[idx])
    if np.abs(1 - val2 / val1) < tol:
        cond = False
        KE_mean = val2
    else:
        val1 = val2
# '''

fln = fln[0:-3]

################################################################################
######################## PLOTTING KE & D (MEAN/MEDIAN) #########################
################################################################################

# '''
fig, ax = plt.subplots(2)

# plt.figure()
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=12)
xmin = 4000
xmax = 7000
plot_mean_std = True # `False` => median and IQR will be plotted

ax[0].plot(t_recorded_samples, D, 'k-', linewidth=1.)
if plot_mean_std == True:
    ax[0].axhline(D_mean)
    ax[0].legend(['D std : {:.4f}'.format(D_std), 'D mean : {:.4f}'.format(D_mean)])
else:
    ax[0].axhline(D_median)
    ax[0].legend(['D IQR : {:.4f}'.format(D_IQR), 'D median : {:.4f}'.format(D_median)])
ax[0].set_xlim(xmin, xmax)
ax[0].set_ylabel(r'$D$')
ax[0].grid(True)

ax[1].plot(t_recorded_samples, K, 'k-', linewidth=1.)
if plot_mean_std == True:
    ax[1].axhline(KE_mean)
    ax[1].legend(['TKE std : {:.4f}'.format(KE_std), 'TKE mean : {:.4f}'.format(KE_mean)])
else:
    ax[1].axhline(KE_median)
    ax[1].legend(['TKE IQR : {:.4f}'.format(KE_IQR), 'TKE median : {:.4f}'.format(KE_median)])
ax[1].set_xlim(xmin, xmax)
ax[1].set_ylim(.5-0.028, .9+0.028)
ax[1].set_ylabel(r'$TKE$')
ax[1].grid(True)


if plot_mean_std == True:
    plt.savefig(fln+'_mean_std.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(fln+'_median_IQR.png', dpi=300, bbox_inches='tight')
# plt.show()
plt.clf()
# '''

################################################################################
############################### MEAN QUANTITIES ################################
################################################################################

fig, ax = plt.subplots(1, 3, figsize=(5.0*(3+0), 5.0*1))

num_xticks = 5
xticks = np.linspace(0, u_mean.shape[0]-1, num_xticks, dtype=np.int32)
xtick_labels = [
    '0',
    r'$\frac{\pi}{2}$',
    r'$\pi$',
    r'$\frac{3\pi}{2}$',
    r'$2\pi$',
]

im_umean = ax[0].imshow(np.flip(u_mean, axis=0), aspect='equal', origin='lower')#, vmin=vmin, vmax=vmax)
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].title.set_text(r'$\bar{u}$')
ax[0].set_xticks(xticks)
ax[0].set_xticklabels(xtick_labels)
ax[0].set_yticks(xticks)
ax[0].set_yticklabels(xtick_labels)
# cbar_u = plt.colorbar(im_umean, ax=ax[0], orientation='vertical')
cb_xbegin = ax[0].transData.transform([0, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax[0].transData.transform([u_mean.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
umean_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_umean = fig.colorbar(im_umean, cax=umean_cb_ax, orientation='horizontal')

im_vmean = ax[1].imshow(np.flip(v_mean, axis=0), aspect='equal', origin='lower')#, vmin=vmin, vmax=vmax)
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$y$')
ax[1].title.set_text(r'$\bar{v}$')
ax[1].set_xticks(xticks)
ax[1].set_xticklabels(xtick_labels)
ax[1].set_yticks(xticks)
ax[1].set_yticklabels(xtick_labels)
# cbar_v = plt.colorbar(im_vmean, ax=ax[1], orientation='vertical')
cb_xbegin = ax[1].transData.transform([0, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax[1].transData.transform([u_mean.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
vmean_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_vmean = fig.colorbar(im_vmean, cax=vmean_cb_ax, orientation='horizontal')

im_vortmean = ax[2].imshow(np.flip(vort_mean, axis=0), aspect='equal', origin='lower')#, vmin=vmin, vmax=vmax)
ax[2].set_xlabel(r'$x$')
ax[2].set_ylabel(r'$y$')
ax[2].title.set_text(r'$\bar{\omega}$')
ax[2].set_xticks(xticks)
ax[2].set_xticklabels(xtick_labels)
ax[2].set_yticks(xticks)
ax[2].set_yticklabels(xtick_labels)
# cbar_vort = plt.colorbar(im_vortmean, ax=ax[2], orientation='vertical')
cb_xbegin = ax[2].transData.transform([0, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax[2].transData.transform([u_mean.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
vortmean_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_vortmean = fig.colorbar(im_vortmean, cax=vortmean_cb_ax, orientation='horizontal')

# plt.tight_layout()
plt.savefig(fln+'_u_v_omega-mean.png', dpi=300, bbox_inches='tight')
# plt.show()

################################################################################
############################# SNAPSHOT QUANTITIES ##############################
################################################################################

fig, ax = plt.subplots(1, 3, figsize=(5.0*(3+0), 5.0*1))

num_xticks = 5
xticks = np.linspace(0, u_snapshot.shape[0]-1, num_xticks, dtype=np.int32)
xtick_labels = [
    '0',
    r'$\frac{\pi}{2}$',
    r'$\pi$',
    r'$\frac{3\pi}{2}$',
    r'$2\pi$',
]

im_usnapshot = ax[0].imshow(np.flip(u_snapshot, axis=0), aspect='equal', origin='lower')#, vmin=vmin, vmax=vmax)
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].title.set_text(r'$u (t = ' + '{:.1f}'.format(t_recorded_samples[snapshot_idx]) + ' \ s.)$')
ax[0].set_xticks(xticks)
ax[0].set_xticklabels(xtick_labels)
ax[0].set_yticks(xticks)
ax[0].set_yticklabels(xtick_labels)
# cbar_u = plt.colorbar(im_usnapshot, ax=ax[0], orientation='horizontal')
cb_xbegin = ax[0].transData.transform([0, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax[0].transData.transform([u_snapshot.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
usnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_usnapshot = fig.colorbar(im_usnapshot, cax=usnapshot_cb_ax, orientation='horizontal')

im_vsnapshot = ax[1].imshow(np.flip(v_snapshot, axis=0), aspect='equal', origin='lower')#, vmin=vmin, vmax=vmax)
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$y$')
ax[1].title.set_text(r'$v (t = ' + '{:.1f}'.format(t_recorded_samples[snapshot_idx]) + ' \ s.)$')
ax[1].set_xticks(xticks)
ax[1].set_xticklabels(xtick_labels)
ax[1].set_yticks(xticks)
ax[1].set_yticklabels(xtick_labels)
# cbar_v = plt.colorbar(im_vsnapshot, ax=ax[1], orientation='horizontal')
cb_xbegin = ax[1].transData.transform([0, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax[1].transData.transform([v_snapshot.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
vsnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_vsnapshot = fig.colorbar(im_vsnapshot, cax=vsnapshot_cb_ax, orientation='horizontal')

im_vortsnapshot = ax[2].imshow(np.flip(vort_snapshot, axis=0), aspect='equal', origin='lower')#, vmin=vmin, vmax=vmax)
ax[2].set_xlabel(r'$x$')
ax[2].set_ylabel(r'$y$')
ax[2].title.set_text(r'$\omega (t = ' + '{:.1f}'.format(t_recorded_samples[snapshot_idx]) + ' \ s.)$')
ax[2].set_xticks(xticks)
ax[2].set_xticklabels(xtick_labels)
ax[2].set_yticks(xticks)
ax[2].set_yticklabels(xtick_labels)
# cbar_vort = plt.colorbar(im_vortsnapshot, ax=ax[2], orientation='horizontal')
cb_xbegin = ax[2].transData.transform([0, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax[2].transData.transform([vort_snapshot.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
vortsnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_vortsnapshot = fig.colorbar(im_vortsnapshot, cax=vortsnapshot_cb_ax, orientation='horizontal')

# plt.tight_layout()
plt.savefig(fln+'_u_v_omega-snapshot.png', dpi=300, bbox_inches='tight')
# plt.show()

################################################################################
