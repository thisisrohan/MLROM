import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import os

# plt.rcParams.update({
#     "text.usetex":True,
#     "font.family":"serif",
# })

"""
Implements a pseudo-spectral solver for the Kolmogorov flow
"""

np.seterr(divide='ignore', invalid='ignore')

class Kol2D_odd(object):
    """
    N: resolution of grid used; number of grids (single direction) = (2N+1)
    Re: Reynolds number
    n: wavernumber of external forcing in x direction
    wave numbers are arranged such that 0 is in the center    
    """
    def __init__(self, Re=40, n=4, N=6):
        
        self.N = N
        self.grid_setup(N)
        self.grids = 2*N + 1
        self.Re = Re
        self.fx = np.fft.fftshift(np.fft.fft2(np.sin(n*self.yy)))

        # aa = np.fft.ifft2(np.fft.ifftshift(self.fx))
        # print(aa.real)
        # print(aa.imag)

    def grid_setup(self, N):
        
        # physical grid
        x = np.linspace(0, 2*np.pi, 2*N+2)
        x = x[:-1]
        self.xx, self.yy = np.meshgrid(x,x)

        # wavenumbers
        k = np.arange(-N, N+1)
        self.kk1, self.kk2 = np.meshgrid(k,k)
        self.kk = self.kk1**2 + self.kk2**2

        # parameters for divergence-free projection (Fourier domain)
        self.p1 = self.kk2**2/self.kk
        self.p2 = -self.kk1*self.kk2/self.kk
        self.p3 = self.kk1**2/self.kk

        # differentiation (Fourier domain)
        self.ddx = 1j*self.kk1
        self.ddy = 1j*self.kk2

        # matrix for converting u,v to a and vice versa: u = a*pu, v = a*pv
        self.pu = self.kk2/np.sqrt(self.kk)
        self.pu[self.N, self.N] = 0
        self.pv = -self.kk1/np.sqrt(self.kk)
        self.pv[self.N, self.N] = 0
        
        self.proj_res_x = np.empty(shape=self.kk.shape, dtype=np.complex128)
        self.proj_res_y = np.empty(shape=self.kk.shape, dtype=np.complex128)
        
        self.dyn_rhs_x = np.empty(shape=self.kk.shape, dtype=np.complex128)
        self.dyn_rhs_y = np.empty(shape=self.kk.shape, dtype=np.complex128)
        
        sz2 = 4*self.N + 1
        self.ff1_h = np.zeros((sz2, sz2), dtype=np.complex128)
        self.ff2_h = np.zeros((sz2, sz2), dtype=np.complex128)
        self.ff1 = np.zeros((sz2, sz2), dtype=np.float64)
        self.ff2 = np.zeros((sz2, sz2), dtype=np.float64)
        self.p_h = np.zeros((2*N+1, 2*N+1), dtype=np.complex128)
        
        

    # @tf.function
    def proj_DF(self, fx_h, fy_h, ux_h, uy_h):    # divergence free projection
        
        ux_h[:, :] = self.p1*fx_h + self.p2*fy_h
        uy_h[:, :] = self.p2*fx_h + self.p3*fy_h

        # boundary conditions
        # if fx_h.ndim == 2:
        #     ux_h[self.N, self.N] = 0
        #     uy_h[self.N, self.N] = 0
        # elif fx_h.ndim == 3:
        #     ux_h[:, self.N, self.N] = 0
        #     uy_h[:, self.N, self.N] = 0
        
        ux_h[self.N, self.N] = 0
        uy_h[self.N, self.N] = 0

        return ux_h, uy_h


    def uv2a(self, u_h, v_h):    # unified Fourier coefficients a(x,t)
        
        a_h = u_h/self.pu
        a_v = v_h/self.pv

        if u_h.ndim == 2:
            a_h[self.N] = a_v[self.N]
            a_h[self.N, self.N] = 0
        elif u_h.ndim == 3:
            a_h[:, self.N, :] = a_v[:, self.N, :]
            a_h[:, self.N, self.N] = 0

        return a_h


    def a2uv(self, a_h):

        return a_h*self.pu, a_h*self.pv


    # @tf.function
    def vort(self, u_h, v_h):        # calculate vorticity
        
        return self.ddy*u_h - self.ddx*v_h


    # @tf.function
    def dissip(self, u_h, v_h):    # calculate dissipation
        
        w_h = self.vort(u_h, v_h)
        D = np.sum(w_h*w_h.conjugate(), axis=(-1,-2))
        D = np.squeeze(D)/self.Re/self.grids**4

        return D.real


    # @tf.function
    def dynamics(self, u_h, v_h, du_h, dv_h):

        self.dyn_rhs_x[:, :] = -self.ddx*self.aap(u_h, u_h, self.ff1_h, self.ff2_h, self.ff1, self.ff2, self.p_h)
        self.dyn_rhs_x[:, :] -= self.ddy*self.aap(u_h, v_h, self.ff1_h, self.ff2_h, self.ff1, self.ff2, self.p_h)
        self.dyn_rhs_x[:, :] += self.fx
        
        self.dyn_rhs_y[:, :] = -self.ddx*self.aap(u_h, v_h, self.ff1_h, self.ff2_h, self.ff1, self.ff2, self.p_h)
        self.dyn_rhs_y[:, :] -= self.ddy*self.aap(v_h, v_h, self.ff1_h, self.ff2_h, self.ff1, self.ff2, self.p_h)

        self.proj_res_x, self.proj_res_y = self.proj_DF(self.dyn_rhs_x, self.dyn_rhs_y, self.proj_res_x, self.proj_res_y)

        du_h[:, :] = -self.kk*u_h/self.Re + self.proj_res_x
        dv_h[:, :] = -self.kk*v_h/self.Re + self.proj_res_y
    
        return du_h, dv_h


    def dynamics_a(self, a_h):
        
        u_h, v_h = self.a2uv(a_h)
        du_h, dv_h = self.dynamics(u_h, v_h)
        da_h = self.uv2a(du_h, dv_h)

        return da_h


    def random_field(self,A_std,A_mag,c1=0,c2=3):

        '''
            generate a random field whose energy is normally distributed
            in Fourier domain centered at wavenumber (c1,c2) with random phase
        '''
        
        A = A_mag * 4 * (self.grids**2) * np.exp(
            -( (self.kk1-c1)**2 + (self.kk2-c2)**2 ) / (2 * A_std**2)
        ) / np.sqrt(2 * np.pi * A_std**2)
        u_h = A*np.exp(1j*2*np.pi*np.random.rand(self.grids, self.grids))
        v_h = A*np.exp(1j*2*np.pi*np.random.rand(self.grids, self.grids))

        u = np.fft.irfft2(np.fft.ifftshift(u_h), s=u_h.shape[-2:])
        v = np.fft.irfft2(np.fft.ifftshift(v_h), s=v_h.shape[-2:])

        u_h = np.fft.fftshift(np.fft.fft2(u))
        v_h = np.fft.fftshift(np.fft.fft2(v))

        self.proj_res_x, self.proj_res_y = self.proj_DF(u_h, v_h, self.proj_res_x, self.proj_res_y)
        
        u_h, v_h = self.proj_res_x.copy(), self.proj_res_y.copy()

        return u_h, v_h


    def plot_vorticity(self,u_h,v_h,wmax=None,subplot=False):
        
        w_h = self.vort(u_h,v_h)
        w = np.fft.ifft2(np.fft.ifftshift(w_h))
        w = w.real

        # calculate color axis limit if not specified
        if not wmax:
            wmax = np.ceil(np.abs(w).max())
        wmin = -wmax

        ## plot with image
        tick_loc = np.array([0,.5,1,1.5,2])*np.pi
        tick_label = ['0','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$']
        im = plt.imshow(w, cmap='RdBu', vmin=wmin, vmax=wmax,
                    extent=[0,2*np.pi,0,2*np.pi],
                    interpolation='spline36',origin='lower')
        plt.xticks(tick_loc,tick_label)
        plt.yticks(tick_loc,tick_label)
        if subplot:
            plt.colorbar(im,fraction=.046,pad=.04)
            plt.tight_layout()
        else:
            plt.colorbar()


    def plot_quiver(self,u_h,v_h):
        
        u = np.fft.ifft2(np.fft.ifftshift(u_h)).real
        v = np.fft.ifft2(np.fft.ifftshift(v_h)).real

        Q = plt.quiver(self.xx, self.yy, u, v, units='width')

        tick_loc = np.array([0,.5,1,1.5,2])*np.pi
        tick_label = ['0','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$']

        plt.xticks(tick_loc,tick_label)
        plt.yticks(tick_loc,tick_label)

    # @tf.function
    def aap(self, f1, f2, ff1_h, ff2_h, ff1, ff2, p_h):        # anti-aliased product

        # ndim = f1.ndim
        # assert ndim < 4, 'input dimensions is greater than 3.'
        # if ndim == 2:
        #     # f1_h, f2_h = np.expand_dims(f1, axis=0).copy(), np.expand_dims(f2, axis=0).copy()
        #     f1_h, f2_h = np.expand_dims(f1.copy(), axis=0), np.expand_dims(f2.copy(), axis=0)
        # elif ndim == 3:
        #     f1_h, f2_h = f1.copy(), f2.copy()
        
        sz2 = 4*self.N + 1
        # ff1_h = np.zeros((f1_h.shape[0], sz2, sz2), dtype=np.complex128)
        # ff2_h = np.zeros((f2_h.shape[0], sz2, sz2), dtype=np.complex128)

        idx1, idx2 = self.N, 3*self.N + 1
        ff1_h[:, :] = 0.0
        ff2_h[:, :] = 0.0
        ff1_h[idx1:idx2, idx1:idx2] = f1
        ff2_h[idx1:idx2, idx1:idx2] = f2

        ff1[:, :] = np.fft.irfft2(np.fft.ifftshift(ff1_h), s=ff1_h.shape[-2:])
        ff2[:, :] = np.fft.irfft2(np.fft.ifftshift(ff2_h), s=ff2_h.shape[-2:])          # must take real part or use irfft2

        ff1_h[:, :] = (sz2/self.grids)**2 * np.fft.fftshift(np.fft.fft2(ff1*ff2))

        p_h[:, :] = ff1_h[idx1:idx2, idx1:idx2]

        # if ndim == 2:
        #     p_h = p_h[0,:,:]

        return p_h


################################################################################
################################ CREATING DATA #################################
################################################################################

operation_mode = 'load_and_continue'
# operation_mode = 'initialize_and_start'

### Kolmogorov flow parameters
if operation_mode == 'initialize_and_start':
    N = 32 # modes (pair)
    n = 4  # forcing wavenumber
    Re = 40.
    T = 50.
    N_ref = 50
    dt = .01       # integration time step
    dTr = 0.25     # recording time step
    T_transient = 10.

    # making data save directory
    dir_name_data = os.getcwd() + '/saved_data' # '/scratch/rkaushik/saved_data'
    if not os.path.isdir(dir_name_data):
        os.makedirs(dir_name_data)

    counter = 0
    while True:
        dir_check = 'data_' + str(counter).zfill(3)
        if os.path.isdir(dir_name_data + '/' + dir_check):
            counter += 1
        else:
            break

    dir_name_data = dir_name_data + '/' + dir_check
    os.makedirs(dir_name_data)
    print('dir_name_data:', dir_name_data)
    fln = dir_name_data + '/'

    ### saving simulation parameters
    data = {
        'num_wavenumbers':2*N+1,
        'num_wavenumber_pairs':N,
        'Re':Re,
        'dt':dt,
        'dTr':dTr,
        'T_transient':T_transient,
        'T':T,
        'N_ref':N_ref,
        'n':n,
    }
    with open(dir_name_data+'/sim_data.txt', 'w') as f:
        f.write(str(data))

    np.savez(
        dir_name_data+'/sim_data',
        num_wavenumbers=2*N+1,
        num_wavenumber_pairs=N,
        Re=Re,
        dt=dt,
        dTr=dTr,
        T_transient=T_transient,
        T=T,
        N_ref=N_ref,
    )

else:
    dir_name_data = os.getcwd() + '/saved_data' # '/scratch/rkaushik/saved_data'
    dir_name_data += '/data_005'

    with np.load(dir_name_data+'/sim_data.npz') as f:
        num_wavenumbers = int(f['num_wavenumbers'])
        N = int(f['num_wavenumber_pairs'])
        Re = float(f['Re'])
        dt = float(f['dt'])
        dTr = float(f['dTr'])
        T_transient = float(f['T_transient'])
        T = float(f['T'])
        N_ref = int(f['N_ref'])
        try:
            n = float(f['n'])
        except:
            n = 4.


num_xticks = 5
xtick_labels = [
    '0',
    r'$\frac{\pi}{2}$',
    r'$\pi$',
    r'$\frac{3\pi}{2}$',
    r'$2\pi$',
]

update_percentages = 5
completion = 0

x = np.linspace(0, 2*np.pi, 2*N+2)
x = x[:-1]
xx, yy = np.meshgrid(x, x)
kol2d = Kol2D_odd(Re, n, N)

num_computed_samples = int((T + 0.5*dt) // dt)
t_computed_samples = np.arange(0, num_computed_samples+1)*dt
recorded_tsteps = int( (dTr + 0.5*dt) // dt ) # the state is recorded every `recorded_tsteps` iterations
t_recorded_samples = np.arange(0, num_computed_samples+1, recorded_tsteps)*dt
num_recorded_samples = len(t_recorded_samples)

uh_recorded = np.zeros((len(t_recorded_samples), 2*N+1, 2*N+1), dtype=np.complex128) # time record of `u_hat` on the actual (2*N+1) grid
vh_recorded = np.zeros((len(t_recorded_samples), 2*N+1, 2*N+1), dtype=np.complex128) # time record of `v_hat` on the actual (2*N+1) grid
D = np.empty(shape=len(t_recorded_samples)) # time record of `dissipation`

if operation_mode =='initialize_and_start':
    ### initial condition: random
    u0h, v0h = kol2d.random_field(A_std=1, A_mag=1e-4, c1=0, c2=n)
    num_recorded_samples_load = 0
else:
    fname = dir_name_data + '/data.h5'
    with h5py.File(fname, 'r') as f:
        t_recorded_samples_load = np.array(f['t'])
        num_recorded_samples_load = len(t_recorded_samples_load)

        uh_recorded[0:num_recorded_samples_load] = np.array(f['uh'])
        # uh_recorded[0:num_recorded_samples_load, :, N:] = np.array(f['uh'])
        # uh_recorded[0:num_recorded_samples_load, 0:N, 0:N] = np.conjugate(uh_recorded[0:num_recorded_samples_load, N+1:, N+1:][:, ::-1, ::-1])
        # uh_recorded[0:num_recorded_samples_load, N+1:, 0:N] = np.conjugate(uh_recorded[0:num_recorded_samples_load, 0:N, N+1:][:, ::-1, ::-1])

        vh_recorded[0:num_recorded_samples_load] = np.array(f['vh'])
        # vh_recorded[0:num_recorded_samples_load, :, N:] = np.array(f['vh'])
        # vh_recorded[0:num_recorded_samples_load, 0:N, 0:N] = np.conjugate(vh_recorded[0:num_recorded_samples_load, N+1:, N+1:][:, ::-1, ::-1])
        # vh_recorded[0:num_recorded_samples_load, N+1:, 0:N] = np.conjugate(vh_recorded[0:num_recorded_samples_load, 0:N, N+1:][:, ::-1, ::-1])

        D[0:num_recorded_samples_load] = np.array(f['Dissipation'])

    u0h = np.empty_like(uh_recorded[-1])
    u0h[:, :] = uh_recorded[-1, :, :]

    v0h = np.empty_like(vh_recorded[-1])
    v0h[:, :] = vh_recorded[-1, :, :]

    u0h_r = np.empty_like(u0h)
    v0h_r = np.empty_like(v0h)

    u0h_r[:, :] = u0h[:, :]
    v0h_r[:, :] = v0h[:, :]

    completion = int(np.round( (num_recorded_samples_load / num_recorded_samples) * (100/update_percentages) )) * update_percentages + update_percentages

### get past transients
t_arr = np.arange(dt, T_transient+0.5*dt, dt)

du0h_1 = np.empty_like(u0h)
du0h_2 = np.empty_like(u0h)
du0h_3 = np.empty_like(u0h)
du0h_4 = np.empty_like(u0h)
dv0h_1 = np.empty_like(v0h)
dv0h_2 = np.empty_like(v0h)
dv0h_3 = np.empty_like(v0h)
dv0h_4 = np.empty_like(v0h)
midRK_uh = np.empty_like(u0h)
midRK_vh = np.empty_like(v0h)

if operation_mode == 'initialize_and_start':
    print('Integrating past initial transients: ')
    completion_tr = 0
    update_percentages = 5
    wtime0 = time.time()
    wtime = wtime0
    for tt in t_arr:
        ### RK4
        du0h_1, dv0h_1 = kol2d.dynamics(u0h, v0h, du0h_1, dv0h_1) # k1
        midRK_uh[:, :] = u0h + 0.5*dt*du0h_1
        midRK_vh[:, :] = v0h + 0.5*dt*dv0h_1
            
        du0h_2, dv0h_2 = kol2d.dynamics(midRK_uh, midRK_vh, du0h_2, dv0h_2) # k2
        midRK_uh[:, :] = u0h + 0.5*dt*du0h_2
        midRK_vh[:, :] = v0h + 0.5*dt*dv0h_2

        du0h_3, dv0h_3 = kol2d.dynamics(midRK_uh, midRK_vh, du0h_3, dv0h_3) # k3
        midRK_uh[:, :] = u0h + dt*du0h_3
        midRK_vh[:, :] = v0h + dt*dv0h_3
        
        du0h_4, dv0h_4 = kol2d.dynamics(midRK_uh, midRK_vh, du0h_4, dv0h_4) # k4
        u0h += dt * (du0h_1 + 2*du0h_2 + 2*du0h_3 + du0h_4) / 6
        v0h += dt * (dv0h_1 + 2*dv0h_2 + 2*dv0h_3 + dv0h_4) / 6
        
        if 100*(tt - t_arr[0]) / (t_arr[-1]-t_arr[0]) >= completion_tr:
            curr_time = time.time()
            print("{:3.1f} % completed, update_time : {:.2f} s., total_time : {:.2f} s.".format(completion_tr, curr_time-wtime, curr_time-wtime0))
            completion_tr += update_percentages
            wtime = curr_time

    u0h_r = u0h.copy()
    v0h_r = v0h.copy()

    ### plotting results at the end of transient phase
    u_temp = np.fft.ifft2(np.fft.ifftshift(u0h_r), s=(N_ref, N_ref)).real
    v_temp = np.fft.ifft2(np.fft.ifftshift(v0h_r), s=(N_ref, N_ref)).real
    vort_temp = kol2d.vort(u0h_r, v0h_r)
    vort_temp = np.fft.ifft2(np.fft.ifftshift(vort_temp), s=(N_ref, N_ref)).real

    ### plotting imshow
    xticks = np.linspace(0, u_temp.shape[0]-1, num_xticks, dtype=np.int32)

    fig, ax = plt.subplots(1, 3, figsize=(5.0*(3+0), 5.0*1))

    im_usnapshot = ax[0].imshow(np.flip(u_temp, axis=0), aspect='equal', origin='lower')#, vmin=vmin, vmax=vmax)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].title.set_text(r'$u (t = ' + '{:.1f}'.format(0) + ' \ s.)$')
    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels(xtick_labels)
    ax[0].set_yticks(xticks)
    ax[0].set_yticklabels(xtick_labels)
    # cbar_u = plt.colorbar(im_usnapshot, ax=ax[0], orientation='horizontal')
    cb_xbegin = ax[0].transData.transform([0, 0])
    cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
    cb_xend = ax[0].transData.transform([u_temp.shape[0], 0])
    cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
    usnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
    cbar_usnapshot = fig.colorbar(im_usnapshot, cax=usnapshot_cb_ax, orientation='horizontal')
    usnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

    im_vsnapshot = ax[1].imshow(np.flip(v_temp, axis=0), aspect='equal', origin='lower')#, vmin=vmin, vmax=vmax)
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$y$')
    ax[1].title.set_text(r'$v (t = ' + '{:.1f}'.format(0) + ' \ s.)$')
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(xtick_labels)
    ax[1].set_yticks(xticks)
    ax[1].set_yticklabels(xtick_labels)
    # cbar_v = plt.colorbar(im_vsnapshot, ax=ax[1], orientation='horizontal')
    cb_xbegin = ax[1].transData.transform([0, 0])
    cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
    cb_xend = ax[1].transData.transform([v_temp.shape[0], 0])
    cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
    vsnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
    cbar_vsnapshot = fig.colorbar(im_vsnapshot, cax=vsnapshot_cb_ax, orientation='horizontal')
    vsnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

    im_vortsnapshot = ax[2].imshow(np.flip(vort_temp, axis=0), aspect='equal', origin='lower')#, vmin=vmin, vmax=vmax)
    ax[2].set_xlabel(r'$x$')
    ax[2].set_ylabel(r'$y$')
    ax[2].title.set_text(r'$\omega (t = ' + '{:.1f}'.format(0) + ' \ s.)$')
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels(xtick_labels)
    ax[2].set_yticks(xticks)
    ax[2].set_yticklabels(xtick_labels)
    # cbar_v = plt.colorbar(im_vsnapshot, ax=ax[1], orientation='horizontal')
    cb_xbegin = ax[2].transData.transform([0, 0])
    cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
    cb_xend = ax[2].transData.transform([vort_temp.shape[0], 0])
    cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
    vortsnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
    cbar_vortsnapshot = fig.colorbar(im_vortsnapshot, cax=vortsnapshot_cb_ax, orientation='horizontal')
    vortsnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

    # plt.tight_layout()
    plt.savefig(fln+'u_v_omega-snapshot-t0.pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()

    ### plotting contourf
    xticks = np.linspace(1, u_temp.shape[0]-1, num_xticks, dtype=np.int32)
    fig, ax = plt.subplots(1, 3, figsize=(5.0*(3+0), 5.0*1))
    levels = 10

    im_usnapshot = ax[0].contourf(np.flip(u_temp, axis=0), origin='lower', levels=levels)#aspect='equal', vmin=vmin, vmax=vmax)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].title.set_text(r'$u (t = ' + '{:.1f}'.format(0) + ' \ s.)$')
    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels(xtick_labels)
    ax[0].set_yticks(xticks)
    ax[0].set_yticklabels(xtick_labels)
    ax[0].set_aspect('equal', 'box')
    # cbar_u = plt.colorbar(im_usnapshot, ax=ax[0], orientation='horizontal')
    cb_xbegin = ax[0].transData.transform([0, 0])
    cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
    cb_xend = ax[0].transData.transform([u_temp.shape[0], 0])
    cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
    usnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
    cbar_usnapshot = fig.colorbar(im_usnapshot, cax=usnapshot_cb_ax, orientation='horizontal')
    usnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

    im_vsnapshot = ax[1].contourf(np.flip(v_temp, axis=0), origin='lower', levels=levels)#aspect='equal', vmin=vmin, vmax=vmax)
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$y$')
    ax[1].title.set_text(r'$v (t = ' + '{:.1f}'.format(0) + ' \ s.)$')
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(xtick_labels)
    ax[1].set_yticks(xticks)
    ax[1].set_yticklabels(xtick_labels)
    ax[1].set_aspect('equal', 'box')
    # cbar_v = plt.colorbar(im_vsnapshot, ax=ax[1], orientation='horizontal')
    cb_xbegin = ax[1].transData.transform([0, 0])
    cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
    cb_xend = ax[1].transData.transform([v_temp.shape[0], 0])
    cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
    vsnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
    cbar_vsnapshot = fig.colorbar(im_vsnapshot, cax=vsnapshot_cb_ax, orientation='horizontal')
    vsnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

    im_vortsnapshot = ax[2].contourf(np.flip(vort_temp, axis=0), origin='lower', levels=levels)#aspect='equal', vmin=vmin, vmax=vmax)
    ax[2].set_xlabel(r'$x$')
    ax[2].set_ylabel(r'$y$')
    ax[2].title.set_text(r'$\omega (t = ' + '{:.1f}'.format(0) + ' \ s.)$')
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels(xtick_labels)
    ax[2].set_yticks(xticks)
    ax[2].set_yticklabels(xtick_labels)
    ax[2].set_aspect('equal', 'box')
    # cbar_v = plt.colorbar(im_vsnapshot, ax=ax[1], orientation='horizontal')
    cb_xbegin = ax[2].transData.transform([0, 0])
    cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
    cb_xend = ax[2].transData.transform([vort_temp.shape[0], 0])
    cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
    vortsnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
    cbar_vortsnapshot = fig.colorbar(im_vortsnapshot, cax=vortsnapshot_cb_ax, orientation='horizontal')
    vortsnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

    # plt.tight_layout()
    plt.savefig(fln+'u_v_omega-snapshot-contourf-t0.pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()


### data generation
# u_recorded = np.zeros((len(t_recorded_samples), 2*N+1, 2*N+1)) # u on the actual fourier (2N+1) grid (not the Nref one)
# v_recorded = np.zeros((len(t_recorded_samples), 2*N+1, 2*N+1)) # v on the actual fourier (2N+1) grid (not the Nref one)
# vort_recorded = np.zeros((len(t_recorded_samples), 2*N+1, 2*N+1)) # vort on the actual fourier (2N+1) grid (not the Nref one)

u_reference_recorded = np.zeros( (len(t_recorded_samples), N_ref, N_ref) ) # u on the reference grid (Nref one)
v_reference_recorded = np.zeros( (len(t_recorded_samples), N_ref, N_ref) ) # v on the reference grid (Nref one)
vort_reference_recorded = np.zeros( (len(t_recorded_samples), N_ref, N_ref) ) # vort on the reference grid (Nref one)

print('Recording state')
iter_t_recorded_samples = 0
if operation_mode == 'load_and_continue':
    iter_t_recorded_samples = num_recorded_samples_load
uh_reference = np.zeros( (N_ref, N_ref), dtype=np.complex128 ) # stores the zero-padded and shifted u_h's
vh_reference = np.zeros( (N_ref, N_ref), dtype=np.complex128 ) # stores the zero-padded and shifted v_h's
vort_h_reference = np.zeros( (N_ref, N_ref), dtype=np.complex128 )  # stores the zero-padded and shifted vort_h'



wtime0 = time.time()
wtime = wtime0

ishift_ref = (N_ref - 2*N)/2
ishift_ref = int(ishift_ref)
if ishift_ref <= 0:
    ishift_ref = 0
    ref_lidx = 0
    ref_ridx = N_ref
    actual_lidx = -ishift_ref
    actual_ridx = -ishift_ref + N_ref
else:
    ref_lidx = ishift_ref
    ref_ridx = ishift_ref + 2*N+1
    actual_lidx = 0
    actual_ridx = 2*N+1


for iter_t_computed_samples in range(1+(num_recorded_samples_load-1)*recorded_tsteps, len(t_computed_samples)):

    if iter_t_computed_samples > 0:
        ### RK4
        du0h_1, dv0h_1 = kol2d.dynamics(u0h, v0h, du0h_1, dv0h_1) # k1
        midRK_uh[:, :] = u0h + 0.5*dt*du0h_1
        midRK_vh[:, :] = v0h + 0.5*dt*dv0h_1
            
        du0h_2, dv0h_2 = kol2d.dynamics(midRK_uh, midRK_vh, du0h_2, dv0h_2) # k2
        midRK_uh[:, :] = u0h + 0.5*dt*du0h_2
        midRK_vh[:, :] = v0h + 0.5*dt*dv0h_2

        du0h_3, dv0h_3 = kol2d.dynamics(midRK_uh, midRK_vh, du0h_3, dv0h_3) # k3
        midRK_uh[:, :] = u0h + dt*du0h_3
        midRK_vh[:, :] = v0h + dt*dv0h_3
        
        du0h_4, dv0h_4 = kol2d.dynamics(midRK_uh, midRK_vh, du0h_4, dv0h_4) # k4
        u0h += dt * (du0h_1 + 2*du0h_2 + 2*du0h_3 + du0h_4) / 6
        v0h += dt * (dv0h_1 + 2*dv0h_2 + 2*dv0h_3 + dv0h_4) / 6

    if t_computed_samples[iter_t_computed_samples] == t_recorded_samples[iter_t_recorded_samples]:
        # u0h_r,v0h_r = u0h.copy(), v0h.copy() # copy for recording purpose
        # u0h_r,v0h_r = u0h, v0h
        u0h_r[:, :] = u0h[:, :]
        v0h_r[:, :] = v0h[:, :]

        d = kol2d.dissip(u0h_r, v0h_r)
        # vort = kol2d.vort(u0h_r, v0h_r)

        # uh_reference[:, :] = 0.0
        # vh_reference[:, :] = 0.0
        # # vort_h_reference[:, :] = 0.0
        # # uh_reference[ishift_ref:ishift+2*N+1, ishift:ishift+2*N+1] = u0h_r
        # # vh_reference[ishift:ishift+2*N+1, ishift:ishift+2*N+1] = v0h_r
        # # vort_h_reference[ishift:ishift+2*N+1, ishift:ishift+2*N+1] = vort
        # uh_reference[ref_lidx:ref_ridx, ref_lidx:ref_ridx] = u0h_r[actual_lidx:actual_ridx, actual_lidx:actual_ridx]
        # vh_reference[ref_lidx:ref_ridx, ref_lidx:ref_ridx] = v0h_r[actual_lidx:actual_ridx, actual_lidx:actual_ridx]
        # vort_h_reference[ref_lidx:ref_ridx, ref_lidx:ref_ridx] = vort[actual_lidx:actual_ridx, actual_lidx:actual_ridx]

        # u_reference_recorded[iter_t_recorded_samples, :, :] = np.fft.ifft2(np.fft.ifftshift(uh_reference)).real
        # v_reference_recorded[iter_t_recorded_samples, :, :] = np.fft.ifft2(np.fft.ifftshift(vh_reference)).real
        # vort_reference_recorded[iter_t_recorded_samples, :, :] = np.fft.ifft2(np.fft.ifftshift(vort_h_reference)).real 

        uh_recorded[iter_t_recorded_samples, :, :] = u0h_r
        vh_recorded[iter_t_recorded_samples, :, :] = v0h_r
        # u_recorded[iter_t_recorded_samples, :, :] = np.fft.ifft2(np.fft.ifftshift(u0h_r)).real
        # v_recorded[iter_t_recorded_samples, :, :] = np.fft.ifft2(np.fft.ifftshift(v0h_r)).real
        # vort_recorded[iter_t_recorded_samples, :, :] = np.fft.ifft2(np.fft.ifftshift(kol2d.vort(u0h_r,v0h_r))).real

        D[iter_t_recorded_samples] = d
        
        iter_t_recorded_samples += 1

    if 100*(iter_t_computed_samples+1) / len(t_computed_samples) >= completion:
        curr_time = time.time()
    
        ### saving the data
        fln = dir_name_data + '/data.h5'
        hf = h5py.File(fln, 'w')
        hf.create_dataset('dt', data=dt)
        hf.create_dataset('dTr', data=dTr)
        hf.create_dataset('t', data=t_recorded_samples[:iter_t_recorded_samples])
        hf.create_dataset('Re', data=Re)
        hf.create_dataset('num_wavenumbers', data=2*N+1)
        hf.create_dataset('num_wavenumbers_pairs', data=N)
        hf.create_dataset('N_ref', data=N_ref)
        hf.create_dataset('kx', data=n)
        hf.create_dataset('x', data=x)
        hf.create_dataset('xx', data=xx)
        hf.create_dataset('yy', data=yy)
        # hf.create_dataset('u_reference', data=u_reference_recorded[:iter_t_recorded_samples])
        # hf.create_dataset('v_reference', data=v_reference_recorded[:iter_t_recorded_samples])
        # hf.create_dataset('vort_reference', data=vort_reference_recorded)
        # hf.create_dataset('u', data=u_recorded)
        # hf.create_dataset('v', data=v_recorded)
        # hf.create_dataset('vort', data=vort_recorded)
        hf.create_dataset('uh', data=uh_recorded[:iter_t_recorded_samples]),#, :, N:])
        hf.create_dataset('vh', data=vh_recorded[:iter_t_recorded_samples]),#, :, N:])
        hf.create_dataset('Dissipation', data=D[:iter_t_recorded_samples])
        hf.close()
        
        print("{} % completed, update_time : {:.2f} s., total_time : {:.2f} s.".format(completion, curr_time-wtime, curr_time-wtime0))
        completion += update_percentages
        wtime = curr_time

################################################################################
################################# READING DATA #################################
################################################################################

for t_iter in range(len(t_recorded_samples)):
    u0h_r[:, :] = uh_recorded[t_iter, :, :]
    v0h_r[:, :] = vh_recorded[t_iter, :, :]

    vort = kol2d.vort(u0h_r, v0h_r)

    uh_reference[:, :] = 0.0
    vh_reference[:, :] = 0.0
    vort_h_reference[:, :] = 0.0
    # uh_reference[ishift_ref:ishift+2*N+1, ishift:ishift+2*N+1] = u0h_r
    # vh_reference[ishift:ishift+2*N+1, ishift:ishift+2*N+1] = v0h_r
    # vort_h_reference[ishift:ishift+2*N+1, ishift:ishift+2*N+1] = vort
    uh_reference[ref_lidx:ref_ridx, ref_lidx:ref_ridx] = u0h_r[actual_lidx:actual_ridx, actual_lidx:actual_ridx]
    vh_reference[ref_lidx:ref_ridx, ref_lidx:ref_ridx] = v0h_r[actual_lidx:actual_ridx, actual_lidx:actual_ridx]
    vort_h_reference[ref_lidx:ref_ridx, ref_lidx:ref_ridx] = vort[actual_lidx:actual_ridx, actual_lidx:actual_ridx]

    # u_reference_recorded[t_iter, :, :] = np.fft.ifft2(np.fft.ifftshift(uh_reference)).real
    # v_reference_recorded[t_iter, :, :] = np.fft.ifft2(np.fft.ifftshift(vh_reference)).real
    # vort_reference_recorded[t_iter, :, :] = np.fft.ifft2(np.fft.ifftshift(vort_h_reference)).real
    
    u_reference_recorded[t_iter, :, :] = np.fft.ifft2(np.fft.ifftshift(u0h_r), s=(N_ref, N_ref)).real
    v_reference_recorded[t_iter, :, :] = np.fft.ifft2(np.fft.ifftshift(v0h_r), s=(N_ref, N_ref)).real
    vort_reference_recorded[t_iter, :, :] = np.fft.ifft2(np.fft.ifftshift(vort), s=(N_ref, N_ref)).real


snapshot_idx = int(t_recorded_samples.shape[0] / 2)

u_ref = u_reference_recorded
u_mean = np.sum(u_ref, axis=0) / u_ref.shape[0]
u_snapshot = u_ref[snapshot_idx].copy()
del(u_ref)
v_ref = v_reference_recorded
v_mean = np.sum(v_ref, axis=0) / v_ref.shape[0]
v_snapshot = v_ref[snapshot_idx].copy()
del(v_ref)
vort_ref = vort_reference_recorded
vort_mean = np.sum(vort_ref, axis=0) / vort_ref.shape[0]
vort_snapshot = vort_ref[snapshot_idx].copy()
del(vort_ref)

# u = u_recorded
# u_mean_org = np.mean(u, axis=0) # np.sum(u, axis=0) / u.shape[0]
# v = v_recorded
# v_mean_org = np.mean(v, axis=0) # np.sum(v, axis=0) / v.shape[0]

T = t_recorded_samples[-1]

################################################################################
######################## COMPUTING REQUIRED QUANTITIES #########################
################################################################################

### computing KE
# K = np.mean((u - u_mean_org)**2 + (v - v_mean_org)**2, axis=-1) # TKE
K = np.mean((u_reference_recorded - u_mean)**2 + (v_reference_recorded - v_mean)**2, axis=-1) # TKE
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
'''
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
'''

fln = dir_name_data + '/'

################################################################################
######################## PLOTTING KE & D (MEAN/MEDIAN) #########################
################################################################################

# '''

# plt.figure()
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=12)
min_time_period_plot = 1000

log10_T = np.log10(T)
e_T = np.ceil(log10_T)
m_T = 10**(log10_T - e_T)
if m_T < 0.5:
    e_T -=1
num_to_round = 10**(e_T-1)

xmin = (4000/12500)*T
xmin = num_to_round * np.round(xmin / num_to_round)
xmax = (7000/12500)*T
xmax = num_to_round * np.round(xmax / num_to_round)

if xmax - xmin < min_time_period_plot:
    xmax = xmin + min_time_period_plot
if xmax > t_recorded_samples[-1]:
    xmax = t_recorded_samples[-1]
if xmax - xmin < min_time_period_plot:
    xmin = t_recorded_samples[0]

plot_mean_std = True # `False` => median and IQR will be plotted

for plot_mean_std in [True, False]:
    fig, ax = plt.subplots(2)
    ax[0].plot(t_recorded_samples, D, 'k-', linewidth=1.)
    if plot_mean_std == True:
        ax[0].axhline(D_mean, linestyle='--', linewidth=0.9)
        ax[0].legend(['D std : {:.4f}'.format(D_std), 'D mean : {:.4f}'.format(D_mean)])
    else:
        ax[0].axhline(D_median, linestyle='--', linewidth=0.9)
        ax[0].legend(['D IQR : {:.4f}'.format(D_IQR), 'D median : {:.4f}'.format(D_median)])
    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylabel(r'$D$')
    ax[0].grid(True)

    ax[1].plot(t_recorded_samples, K, 'k-', linewidth=1.)
    if plot_mean_std == True:
        ax[1].axhline(KE_mean, linestyle='--', linewidth=0.9)
        ax[1].legend(['TKE std : {:.4f}'.format(KE_std), 'TKE mean : {:.4f}'.format(KE_mean)])
    else:
        ax[1].axhline(KE_median, linestyle='--', linewidth=0.9)
        ax[1].legend(['TKE IQR : {:.4f}'.format(KE_IQR), 'TKE median : {:.4f}'.format(KE_median)])
    ax[1].set_xlim(xmin, xmax)
    # ax[1].set_ylim(.5-0.028, .9+0.028)
    ax[1].set_ylabel(r'$TKE$')
    ax[1].grid(True)


    if plot_mean_std == True:
        plt.savefig(fln+'mean_std.pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(fln+'median_IQR.pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()
# '''

################################################################################
############################### MEAN QUANTITIES ################################
################################################################################

### plotting imshow
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
umean_cb_ax.tick_params(axis='x', rotation=270+45)

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
vmean_cb_ax.tick_params(axis='x', rotation=270+45)

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
vortmean_cb_ax.tick_params(axis='x', rotation=270+45)

# plt.tight_layout()
plt.savefig(fln+'u_v_omega-mean.pdf', dpi=300, bbox_inches='tight')
# plt.show()

### plotting contourf
xticks = np.linspace(1, u_mean.shape[0]-1, num_xticks, dtype=np.int32)
fig, ax = plt.subplots(1, 3, figsize=(5.0*(3+0), 5.0*1))
levels = 10

im_usnapshot = ax[0].contourf(np.flip(u_mean, axis=0), origin='lower', levels=levels)#aspect='equal', vmin=vmin, vmax=vmax)
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].title.set_text(r'$\bar{u}$')
ax[0].set_xticks(xticks)
ax[0].set_xticklabels(xtick_labels)
ax[0].set_yticks(xticks)
ax[0].set_yticklabels(xtick_labels)
ax[0].set_aspect('equal', 'box')
# cbar_u = plt.colorbar(im_usnapshot, ax=ax[0], orientation='horizontal')
cb_xbegin = ax[0].transData.transform([0, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax[0].transData.transform([u_mean.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
usnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_usnapshot = fig.colorbar(im_usnapshot, cax=usnapshot_cb_ax, orientation='horizontal')
usnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

im_vsnapshot = ax[1].contourf(np.flip(v_mean, axis=0), origin='lower', levels=levels)#aspect='equal', vmin=vmin, vmax=vmax)
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$y$')
ax[1].title.set_text(r'$\bar{v}$')
ax[1].set_xticks(xticks)
ax[1].set_xticklabels(xtick_labels)
ax[1].set_yticks(xticks)
ax[1].set_yticklabels(xtick_labels)
ax[1].set_aspect('equal', 'box')
# cbar_v = plt.colorbar(im_vsnapshot, ax=ax[1], orientation='horizontal')
cb_xbegin = ax[1].transData.transform([0, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax[1].transData.transform([v_mean.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
vsnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_vsnapshot = fig.colorbar(im_vsnapshot, cax=vsnapshot_cb_ax, orientation='horizontal')
vsnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

im_vortsnapshot = ax[2].contourf(np.flip(vort_mean, axis=0), origin='lower', levels=levels)#aspect='equal', vmin=vmin, vmax=vmax)
ax[2].set_xlabel(r'$x$')
ax[2].set_ylabel(r'$y$')
ax[2].title.set_text(r'$\bar{\omega}$')
ax[2].set_xticks(xticks)
ax[2].set_xticklabels(xtick_labels)
ax[2].set_yticks(xticks)
ax[2].set_yticklabels(xtick_labels)
ax[2].set_aspect('equal', 'box')
# cbar_v = plt.colorbar(im_vsnapshot, ax=ax[1], orientation='horizontal')
cb_xbegin = ax[2].transData.transform([0, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax[2].transData.transform([v_mean.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
vortsnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_vortsnapshot = fig.colorbar(im_vortsnapshot, cax=vortsnapshot_cb_ax, orientation='horizontal')
vortsnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

# plt.tight_layout()
plt.savefig(fln+'u_v_omega-mean-contourf.pdf', dpi=300, bbox_inches='tight')
# plt.show()
plt.clf()

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
usnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

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
vsnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

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
vortsnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

# plt.tight_layout()
plt.savefig(fln+'u_v_omega-snapshot.pdf', dpi=300, bbox_inches='tight')
# plt.show()

### plotting contourf
xticks = np.linspace(1, u_snapshot.shape[0]-1, num_xticks, dtype=np.int32)
fig, ax = plt.subplots(1, 3, figsize=(5.0*(3+0), 5.0*1))
levels = 10

im_usnapshot = ax[0].contourf(np.flip(u_snapshot, axis=0), origin='lower', levels=levels)#aspect='equal', vmin=vmin, vmax=vmax)
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].title.set_text(r'$u (t = ' + '{:.1f}'.format(t_recorded_samples[snapshot_idx]) + ' \ s.)$')
ax[0].set_xticks(xticks)
ax[0].set_xticklabels(xtick_labels)
ax[0].set_yticks(xticks)
ax[0].set_yticklabels(xtick_labels)
ax[0].set_aspect('equal', 'box')
# cbar_u = plt.colorbar(im_usnapshot, ax=ax[0], orientation='horizontal')
cb_xbegin = ax[0].transData.transform([0, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax[0].transData.transform([u_snapshot.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
usnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_usnapshot = fig.colorbar(im_usnapshot, cax=usnapshot_cb_ax, orientation='horizontal')
usnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

im_vsnapshot = ax[1].contourf(np.flip(v_snapshot, axis=0), origin='lower', levels=levels)#aspect='equal', vmin=vmin, vmax=vmax)
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$y$')
ax[1].title.set_text(r'$v (t = ' + '{:.1f}'.format(t_recorded_samples[snapshot_idx]) + ' \ s.)$')
ax[1].set_xticks(xticks)
ax[1].set_xticklabels(xtick_labels)
ax[1].set_yticks(xticks)
ax[1].set_yticklabels(xtick_labels)
ax[1].set_aspect('equal', 'box')
# cbar_v = plt.colorbar(im_vsnapshot, ax=ax[1], orientation='horizontal')
cb_xbegin = ax[1].transData.transform([0, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax[1].transData.transform([v_snapshot.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
vsnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_vsnapshot = fig.colorbar(im_vsnapshot, cax=vsnapshot_cb_ax, orientation='horizontal')
vsnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

im_vortsnapshot = ax[2].contourf(np.flip(vort_snapshot, axis=0), origin='lower', levels=levels)#aspect='equal', vmin=vmin, vmax=vmax)
ax[2].set_xlabel(r'$x$')
ax[2].set_ylabel(r'$y$')
ax[2].title.set_text(r'$\omega (t = ' + '{:.1f}'.format(t_recorded_samples[snapshot_idx]) + ' \ s.)$')
ax[2].set_xticks(xticks)
ax[2].set_xticklabels(xtick_labels)
ax[2].set_yticks(xticks)
ax[2].set_yticklabels(xtick_labels)
ax[2].set_aspect('equal', 'box')
# cbar_v = plt.colorbar(im_vsnapshot, ax=ax[1], orientation='horizontal')
cb_xbegin = ax[2].transData.transform([0, 0])
cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
cb_xend = ax[2].transData.transform([vort_snapshot.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
vortsnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_vortsnapshot = fig.colorbar(im_vortsnapshot, cax=vortsnapshot_cb_ax, orientation='horizontal')
vortsnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

# plt.tight_layout()
plt.savefig(fln+'u_v_omega-snapshot-contourf.pdf', dpi=300, bbox_inches='tight')
# plt.show()
plt.clf()

plt.close('all')

################################################################################

