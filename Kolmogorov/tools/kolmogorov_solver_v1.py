import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import tensorflow as tf

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

    # @tf.function
    def proj_DF(self, fx_h, fy_h):    # divergence free projection
        
        ux_h = self.p1*fx_h + self.p2*fy_h
        uy_h = self.p2*fx_h + self.p3*fy_h

        # boundary conditions
        if fx_h.ndim == 2:
            ux_h[self.N, self.N] = 0
            uy_h[self.N, self.N] = 0

        elif fx_h.ndim == 3:
            ux_h[:, self.N, self.N] = 0
            uy_h[:, self.N, self.N] = 0

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
    def vort(self,u_h,v_h):        # calculate vorticity
        
        return self.ddy*u_h - self.ddx*v_h


    # @tf.function
    def dissip(self,u_h,v_h):    # calculate dissipation
        
        w_h = self.vort(u_h,v_h)
        D = np.sum(w_h*w_h.conjugate(),axis=(-1,-2))
        D = np.squeeze(D)/self.Re/self.grids**4

        return D.real


    # @tf.function
    def dynamics(self,u_h,v_h):

        fx_h = -self.ddx*self.aap(u_h,u_h) - self.ddy*self.aap(u_h,v_h) + self.fx
        fy_h = -self.ddx*self.aap(u_h,v_h) - self.ddy*self.aap(v_h,v_h)

        Pfx_h,Pfy_h = self.proj_DF(fx_h,fy_h)

        du_h = -self.kk*u_h/self.Re + Pfx_h
        dv_h = -self.kk*v_h/self.Re + Pfy_h
    
        return du_h,dv_h


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

        u_h,v_h = self.proj_DF(u_h,v_h)

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
    def aap(self,f1,f2):        # anti-aliased product

        ndim = f1.ndim
        assert ndim < 4, 'input dimensions is greater than 3.'
        if ndim == 2:
            # f1_h, f2_h = np.expand_dims(f1, axis=0).copy(), np.expand_dims(f2, axis=0).copy()
            f1_h, f2_h = np.expand_dims(f1.copy(), axis=0), np.expand_dims(f2.copy(), axis=0)
        elif ndim == 3:
            f1_h, f2_h = f1.copy(), f2.copy()
        
        sz2 = 4*self.N + 1
        ff1_h = np.zeros((f1_h.shape[0], sz2, sz2), dtype=np.complex128)
        ff2_h = np.zeros((f2_h.shape[0], sz2, sz2), dtype=np.complex128)

        idx1, idx2 = self.N, 3*self.N + 1
        ff1_h[:, idx1:idx2, idx1:idx2] = f1_h
        ff2_h[:, idx1:idx2, idx1:idx2] = f2_h

        ff1 = np.fft.irfft2(np.fft.ifftshift(ff1_h), s=ff1_h.shape[-2:])
        ff2 = np.fft.irfft2(np.fft.ifftshift(ff2_h), s=ff2_h.shape[-2:])          # must take real part or use irfft2

        pp_h = (sz2/self.grids)**2 * np.fft.fft2(ff1*ff2)
        pp_h = np.fft.fftshift(pp_h)

        p_h = pp_h[:, idx1:idx2, idx1:idx2]

        if ndim == 2:
            p_h = p_h[0,:,:]

        return p_h


################################################################################
################################ CREATING DATA #################################
################################################################################

### Kolmogorov flow parameters
N = 24 # modes (pair)
n = 4  # forcing wavenumber
Re = 40.
T = 12500.

x = np.linspace(0, 2*np.pi, 2*N+2)
x = x[:-1]
xx, yy = np.meshgrid(x,x)
kol2d = Kol2D_odd(Re, n, N)    

### initial condition: random
u0h, v0h = kol2d.random_field(A_std=1, A_mag=1e-4, c1=0, c2=n)

dt = .01       # integration time step
dTr = 0.25       # recording time step
recorded_tsteps = int( (dTr + 0.5*dt) // dt ) # the state is recorded every `recorded_tsteps` iterations

### get past transients
T_transient = 50.
t_arr = np.arange(dt, T_transient+0.5*dt, dt)
print('Integrating past initial transients: ')
completion = 0
update_percentages = 5
wtime0 = time.time()
wtime = wtime0
current_guess_uh = u0h.copy()
current_guess_vh = v0h.copy()
current_guess_duh = np.empty_like(u0h)
current_guess_dvh = np.empty_like(v0h)
phi_uh = np.empty_like(u0h)
phi_vh = np.empty_like(v0h)
updated_guess_uh = np.empty_like(u0h)
updated_guess_vh = np.empty_like(v0h)
for tt in t_arr:
    ### forward Euler
    # du0h, dv0h = kol2d.dynamics(u0h, v0h)
    # u0h += dt*du0h
    # v0h += dt*dv0h
    ### RK4
    du0h_1, dv0h_1 = kol2d.dynamics(u0h, v0h) # k1
    du0h_2, dv0h_2 = kol2d.dynamics(u0h + 0.5*dt*du0h_1, v0h + 0.5*dt*dv0h_1) # k2
    du0h_3, dv0h_3 = kol2d.dynamics(u0h + 0.5*dt*du0h_2, v0h + 0.5*dt*dv0h_2) # k3
    du0h_4, dv0h_4 = kol2d.dynamics(u0h + dt*du0h_3, v0h + dt*dv0h_3) # k4
    u0h += dt * (du0h_1 + 2*du0h_2 + 2*du0h_3 + du0h_4) / 6
    v0h += dt * (dv0h_1 + 2*dv0h_2 + 2*dv0h_3 + dv0h_4) / 6
    ### Crank-Nicholson
    # tol = 1e-6
    # alpha = 4/8
    # cond = True
    # du0h, dv0h = kol2d.dynamics(u0h, v0h)
    # # current_guess_uh = u0h
    # # current_guess_vh = v0h
    # while cond == True:
    #     current_guess_duh[:, :], current_guess_dvh[:, :] = kol2d.dynamics(current_guess_uh, current_guess_vh)
    #     phi_uh[:, :] = u0h + 0.5*dt*(du0h + current_guess_duh)
    #     phi_vh[:, :] = v0h + 0.5*dt*(dv0h + current_guess_dvh)
    #     # performing update
    #     updated_guess_uh[:, :] = alpha * current_guess_uh + (1-alpha)*phi_uh
    #     updated_guess_vh[:, :] = alpha * current_guess_vh + (1-alpha)*phi_vh
    #     iteration_error = np.mean(np.abs(updated_guess_uh - current_guess_uh)**2 + np.abs(updated_guess_vh - current_guess_vh)**2)
    #     if iteration_error <= tol:
    #         cond = False
    #     current_guess_uh[:, :] = updated_guess_uh
    #     current_guess_vh[:, :] = updated_guess_vh
    # u0h[:, :] = current_guess_uh[:, :]
    # v0h[:, :] = current_guess_vh[:, :]
    ### backward Euler
    # tol = 1e-3
    # alpha = 7/8
    # cond = True
    # du0h, dv0h = kol2d.dynamics(u0h, v0h)
    # current_guess_uh = u0h
    # current_guess_vh = v0h
    # while cond == True:
    #     current_guess_duh, current_guess_dvh = kol2d.dynamics(current_guess_uh, current_guess_vh)
    #     phi_uh = u0h + dt*current_guess_duh
    #     phi_vh = v0h + dt*current_guess_dvh
    #     # performing update
    #     updated_guess_uh = alpha * current_guess_uh + (1-alpha)*phi_uh
    #     updated_guess_vh = alpha * current_guess_vh + (1-alpha)*phi_vh
    #     iteration_error = np.mean(np.abs(updated_guess_uh - current_guess_uh)**2 + np.abs(updated_guess_vh - current_guess_vh)**2)**0.5
    #     if iteration_error <= tol:
    #         cond = False
    #     current_guess_uh = updated_guess_uh
    #     current_guess_vh = updated_guess_vh
    # u0h[:, :] = current_guess_uh[:, :]
    # v0h[:, :] = current_guess_vh[:, :]
    
    if 100*(tt - t_arr[0]) / (t_arr[-1]-t_arr[0]) >= completion:
        curr_time = time.time()
        print("{:3.1f} % completed, update_time : {:.2f} s., total_time : {:.2f} s.".format(completion, curr_time-wtime, curr_time-wtime0))
        completion += update_percentages
        wtime = curr_time
u0h_r = u0h.copy()
v0h_r = v0h.copy()

u_temp = np.fft.ifft2(np.fft.ifftshift(u0h_r)).real
v_temp = np.fft.ifft2(np.fft.ifftshift(v0h_r)).real
vort_temp = kol2d.vort(u0h_r, v0h_r)
vort_temp = np.fft.ifft2(np.fft.ifftshift(vort_temp)).real

num_xticks = 5
xticks = np.linspace(0, u_temp.shape[0]-1, num_xticks, dtype=np.int32)
xtick_labels = [
    '0',
    r'$\frac{\pi}{2}$',
    r'$\pi$',
    r'$\frac{3\pi}{2}$',
    r'$2\pi$',
]

### plotting imshow
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
fln = 'Kolmogorov_Re' + str(Re) + '_T' + str(T) + '_DT01'
plt.savefig(fln+'_u_v_omega-snapshot-t0.png', dpi=300, bbox_inches='tight')
# plt.show()
plt.clf()

### plotting contourf
xticks = np.linspace(1, u_temp.shape[0]-1, num_xticks, dtype=np.int32)
fig, ax = plt.subplots(1, 3, figsize=(5.0*(3+0), 5.0*1))
levels = 11

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
fln = 'Kolmogorov_Re' + str(Re) + '_T' + str(T) + '_DT01'
plt.savefig(fln+'_u_v_omega-snapshot-contourf-t0.png', dpi=300, bbox_inches='tight')
# plt.show()
plt.clf()

### data generation
num_computed_samples = int((T + 0.5*dt) // dt)
t_computed_samples = np.arange(0, num_computed_samples+1)*dt
t_recorded_samples = np.arange(0, num_computed_samples+1, recorded_tsteps)*dt
num_recorded_samples = len(t_recorded_samples)

Uh = np.expand_dims(u0h_r, axis=0)
Vh = np.expand_dims(v0h_r, axis=0)
# dUh,dVh = np.expand_dims(du0h,axis=0),np.expand_dims(dv0h,axis=0)

uh_recorded = np.zeros((len(t_recorded_samples), 2*N+1, 2*N+1), dtype=np.complex128)
vh_recorded = np.zeros((len(t_recorded_samples), 2*N+1, 2*N+1), dtype=np.complex128)
u_recorded = np.zeros((len(t_recorded_samples), 2*N+1, 2*N+1))
v_recorded = np.zeros((len(t_recorded_samples), 2*N+1, 2*N+1))
vort_recorded = np.zeros((len(t_recorded_samples), 2*N+1, 2*N+1))
D = np.empty(shape=len(t_recorded_samples))

N_ref = 50
u_reference_recorded = np.zeros( (len(t_recorded_samples), N_ref, N_ref) )
v_reference_recorded = np.zeros( (len(t_recorded_samples), N_ref, N_ref) )
vort_reference_recorded = np.zeros( (len(t_recorded_samples), N_ref, N_ref) )

print('Recording state')
iter_t_recorded_samples = 0
uh_reference = np.zeros( (N_ref, N_ref), dtype=np.complex128 )
vh_reference = np.zeros( (N_ref, N_ref), dtype=np.complex128 )
vort_h_reference = np.zeros( (N_ref, N_ref), dtype=np.complex128 )
u0h_r = np.empty_like(u0h)
v0h_r = np.empty_like(v0h)

update_percentages = 5
completion = 0
wtime0 = time.time()
wtime = wtime0
current_guess_uh[:, :] = u0h
current_guess_vh[:, :] = v0h
for iter_t_computed_samples in range(len(t_computed_samples)):

    if iter_t_computed_samples > 0:
        ### forward Euler
        # du0h, dv0h = kol2d.dynamics(u0h, v0h)
        # u0h += dt*du0h
        # v0h += dt*dv0h
        ### RK4
        # du0h_1, dv0h_1 = kol2d.dynamics(u0h, v0h) # k1
        # du0h_2, dv0h_2 = kol2d.dynamics(u0h + 0.5*dt*du0h_1, v0h + 0.5*dt*dv0h_1) # k2
        # du0h_3, dv0h_3 = kol2d.dynamics(u0h + 0.5*dt*du0h_2, v0h + 0.5*dt*dv0h_2) # k3
        # du0h_4, dv0h_4 = kol2d.dynamics(u0h + dt*du0h_3, v0h + dt*dv0h_3) # k4
        # u0h += dt * (du0h_1 + 2*du0h_2 + 2*du0h_3 + du0h_4) / 6
        # v0h += dt * (dv0h_1 + 2*dv0h_2 + 2*dv0h_3 + dv0h_4) / 6
        ### Crank-Nicholson
        tol = 1e-6
        alpha = 4/8
        cond = True
        du0h, dv0h = kol2d.dynamics(u0h, v0h)
        # current_guess_uh = u0h
        # current_guess_vh = v0h
        while cond == True:
            current_guess_duh[:, :], current_guess_dvh[:, :] = kol2d.dynamics(current_guess_uh, current_guess_vh)
            phi_uh[:, :] = u0h + 0.5*dt*(du0h + current_guess_duh)
            phi_vh[:, :] = v0h + 0.5*dt*(dv0h + current_guess_dvh)
            # performing update
            updated_guess_uh[:, :] = alpha * current_guess_uh + (1-alpha)*phi_uh
            updated_guess_vh[:, :] = alpha * current_guess_vh + (1-alpha)*phi_vh
            iteration_error = np.mean(np.abs(updated_guess_uh - current_guess_uh)**2 + np.abs(updated_guess_vh - current_guess_vh)**2)
            if iteration_error <= tol:
                cond = False
            current_guess_uh[:, :] = updated_guess_uh
            current_guess_vh[:, :] = updated_guess_vh
        u0h[:, :] = current_guess_uh[:, :]
        v0h[:, :] = current_guess_vh[:, :]
        ### backward Euler
        # tol = 1e-3
        # alpha = 7/8
        # cond = True
        # du0h, dv0h = kol2d.dynamics(u0h, v0h)
        # current_guess_uh = u0h
        # current_guess_vh = v0h
        # while cond == True:
        #     current_guess_duh, current_guess_dvh = kol2d.dynamics(current_guess_uh, current_guess_vh)
        #     phi_uh = u0h + dt*current_guess_duh
        #     phi_vh = v0h + dt*current_guess_dvh
        #     # performing update
        #     updated_guess_uh = alpha * current_guess_uh + (1-alpha)*phi_uh
        #     updated_guess_vh = alpha * current_guess_vh + (1-alpha)*phi_vh
        #     iteration_error = np.mean(np.abs(updated_guess_uh - current_guess_uh)**2 + np.abs(updated_guess_vh - current_guess_vh)**2)**0.5
        #     if iteration_error <= tol:
        #         cond = False
        #     current_guess_uh = updated_guess_uh
        #     current_guess_vh = updated_guess_vh
        # u0h[:, :] = current_guess_uh[:, :]
        # v0h[:, :] = current_guess_vh[:, :]

    if t_computed_samples[iter_t_computed_samples] == t_recorded_samples[iter_t_recorded_samples]:
        # u0h_r,v0h_r = u0h.copy(), v0h.copy() # copy for recording purpose
        # u0h_r,v0h_r = u0h, v0h
        u0h_r[:, :] = u0h[:, :]
        v0h_r[:, :] = v0h[:, :]

        d = kol2d.dissip(u0h_r, v0h_r)
        vort = kol2d.vort(u0h_r, v0h_r)

        uh_reference[:, :] = 0.0
        vh_reference[:, :] = 0.0
        vort_h_reference[:, :] = 0.0
        ishift = (N_ref - 2*N)/2
        ishift = int(ishift)
        uh_reference[ishift:ishift+2*N+1, ishift:ishift+2*N+1] = u0h_r
        vh_reference[ishift:ishift+2*N+1, ishift:ishift+2*N+1] = v0h_r
        vort_h_reference[ishift:ishift+2*N+1, ishift:ishift+2*N+1] = vort

        u_reference_recorded[iter_t_recorded_samples, :, :] = np.fft.ifft2(np.fft.ifftshift(uh_reference)).real
        v_reference_recorded[iter_t_recorded_samples, :, :] = np.fft.ifft2(np.fft.ifftshift(vh_reference)).real
        vort_reference_recorded[iter_t_recorded_samples, :, :] = np.fft.ifft2(np.fft.ifftshift(vort_h_reference)).real 

        uh_recorded[iter_t_recorded_samples, :, :] = u0h_r
        vh_recorded[iter_t_recorded_samples, :, :] = v0h_r
        u_recorded[iter_t_recorded_samples, :, :] = np.fft.ifft2(np.fft.ifftshift(u0h_r)).real
        v_recorded[iter_t_recorded_samples, :, :] = np.fft.ifft2(np.fft.ifftshift(v0h_r)).real
        vort_recorded[iter_t_recorded_samples, :, :] = np.fft.ifft2(np.fft.ifftshift(kol2d.vort(u0h_r,v0h_r))).real
     
        #Tr.append(tt1)
        #Uh = np.r_[Uh,[u0h_r]]        # record state and dynamics
        #Vh = np.r_[Vh,[v0h_r]]

        # dUh = np.r_[dUh,[du0h]]
        # dVh = np.r_[dVh,[dv0h]]
        D[iter_t_recorded_samples] = d
        
        iter_t_recorded_samples += 1

    if 100*(iter_t_computed_samples+1) / len(t_computed_samples) >= completion:
        curr_time = time.time()
        print("{} % completed, update_time : {:.2f} s., total_time : {:.2f} s.".format(completion, curr_time-wtime, curr_time-wtime0))
        completion += update_percentages
        wtime = curr_time
    

### saving the data
fln = 'Kolmogorov_Re' + str(Re) + '_T' + str(T) + '_dt' + str(dt) + '_N' + str(N) + '.h5'
hf = h5py.File(fln, 'w')
hf.create_dataset('dt', data=dt)
hf.create_dataset('dTr', data=dTr)
hf.create_dataset('t', data=t_recorded_samples)
hf.create_dataset('Re', data=Re)
hf.create_dataset('num_wavenumbers', data=2*N+1)
hf.create_dataset('N_ref', data=N_ref)
hf.create_dataset('kx', data=n)
hf.create_dataset('x', data=x)
hf.create_dataset('xx', data=xx)
hf.create_dataset('yy', data=yy)
# hf.create_dataset('u_reference', data=u_reference_recorded)
# hf.create_dataset('v_reference', data=v_reference_recorded)
# hf.create_dataset('vort_reference', data=vort_reference_recorded)
# hf.create_dataset('u', data=u_recorded)
# hf.create_dataset('v', data=v_recorded)
# hf.create_dataset('vort', data=vort_recorded)
hf.create_dataset('uh', data=uh_recorded)
hf.create_dataset('vh', data=vh_recorded)
hf.create_dataset('Dissipation', data=D)
hf.close()

################################################################################
################################# READING DATA #################################
################################################################################

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

u = u_recorded
u_mean_org = np.mean(u, axis=0) # np.sum(u, axis=0) / u.shape[0]
v = v_recorded
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

fln = fln[0:-3]

################################################################################
######################## PLOTTING KE & D (MEAN/MEDIAN) #########################
################################################################################

# '''

# plt.figure()
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=12)
xmin = 4000
xmax = 7000
plot_mean_std = True # `False` => median and IQR will be plotted

for plot_mean_std in [True, False]:
    fig, ax = plt.subplots(2)
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
plt.savefig(fln+'_u_v_omega-mean.png', dpi=300, bbox_inches='tight')
# plt.show()

### plotting contourf
xticks = np.linspace(1, u_mean.shape[0]-1, num_xticks, dtype=np.int32)
fig, ax = plt.subplots(1, 3, figsize=(5.0*(3+0), 5.0*1))
levels = 11

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
cb_xend = ax[0].transData.transform([u_temp.shape[0], 0])
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
cb_xend = ax[1].transData.transform([v_temp.shape[0], 0])
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
cb_xend = ax[2].transData.transform([vort_temp.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
vortsnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_vortsnapshot = fig.colorbar(im_vortsnapshot, cax=vortsnapshot_cb_ax, orientation='horizontal')
vortsnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

# plt.tight_layout()
fln = 'Kolmogorov_Re' + str(Re) + '_T' + str(T) + '_DT01'
plt.savefig(fln+'_u_v_omega-mean-contourf.png', dpi=300, bbox_inches='tight')
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
plt.savefig(fln+'_u_v_omega-snapshot.png', dpi=300, bbox_inches='tight')
# plt.show()

### plotting contourf
xticks = np.linspace(1, u_snapshot.shape[0]-1, num_xticks, dtype=np.int32)
fig, ax = plt.subplots(1, 3, figsize=(5.0*(3+0), 5.0*1))
levels = 11

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
cb_xend = ax[0].transData.transform([u_temp.shape[0], 0])
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
cb_xend = ax[1].transData.transform([v_temp.shape[0], 0])
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
cb_xend = ax[2].transData.transform([vort_temp.shape[0], 0])
cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]
vortsnapshot_cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
cbar_vortsnapshot = fig.colorbar(im_vortsnapshot, cax=vortsnapshot_cb_ax, orientation='horizontal')
vortsnapshot_cb_ax.tick_params(axis='x', rotation=270+45)

# plt.tight_layout()
fln = 'Kolmogorov_Re' + str(Re) + '_T' + str(T) + '_DT01'
plt.savefig(fln+'_u_v_omega-snapshot-contourf.png', dpi=300, bbox_inches='tight')
# plt.show()
plt.clf()

################################################################################

