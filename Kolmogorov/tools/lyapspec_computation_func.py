import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import tensorflow as tf
import os
import scipy.fft as spfftmodule

"""
Implements a pseudo-spectral solver for the Kolmogorov flow, creates functions
for lyapunov spectrum computation as well
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
########################### CREATING DATA FUNCTIONS ############################
################################################################################


def create_Kol_data_for_lyap(
        init_state_mat,
        params_mat=[40],
        t0=0., T=5., delta_t=0.01,
        N_ref=50, N=24, n=4,
        A_std=1, A_mag=1e-4,        
        kol2D_obj=None,
        u0h=None, v0h=None, du0h_1=None, dv0h_1=None,
        midRK_uh=None, midRK_vh=None, du0h_2=None, dv0h_2=None,
        du0h_3=None, dv0h_3=None,
        du0h_4=None, dv0h_4=None,
        uh_reference=None, vh_reference=None,
        **kwargs,
    ):
    
    Re = params_mat[0]
    dt = delta_t
    if type(kol2D_obj) == type(None):
        kol2D_obj = Kol2D_odd(Re, n, N)
        
    if type(u0h) == type(None) or type(v0h) == type(None):
        u0h = np.empty(shape=(2*N+1, 2*N+1), dtype=np.complex128)
        v0h = np.empty(shape=(2*N+1, 2*N+1), dtype=np.complex128)

    ishift = (N_ref - 2*N)/2
    ishift = int(ishift)
    
    init_state_mat = np.reshape(init_state_mat, (2, N_ref, N_ref))
    temp_u0h = np.fft.fft2(init_state_mat[0])
    temp_u0h = np.fft.fftshift(temp_u0h)
    u0h[:, :] = temp_u0h[ishift:ishift+2*N+1, ishift:ishift+2*N+1]
    del(temp_u0h)
    temp_v0h = np.fft.fft2(init_state_mat[1])
    temp_v0h = np.fft.fftshift(temp_v0h)
    v0h[:, :] = temp_v0h[ishift:ishift+2*N+1, ishift:ishift+2*N+1]
    del(temp_v0h)

    T = T - t0
    num_computed_samples = int((T + 0.5*dt) // dt)
    t_computed_samples = np.arange(0, num_computed_samples+1)*dt
    
    if type(du0h_1) == type(None): 
        du0h_1 = np.empty_like(u0h)
    if type(du0h_2) == type(None): 
        du0h_2 = np.empty_like(u0h)
    if type(du0h_3) == type(None): 
        du0h_3 = np.empty_like(u0h)
    if type(du0h_4) == type(None): 
        du0h_4 = np.empty_like(u0h)
    if type(dv0h_1) == type(None): 
        dv0h_1 = np.empty_like(v0h)
    if type(dv0h_2) == type(None): 
        dv0h_2 = np.empty_like(v0h)
    if type(dv0h_3) == type(None): 
        dv0h_3 = np.empty_like(v0h)
    if type(dv0h_4) == type(None): 
        dv0h_4 = np.empty_like(v0h)
    if type(midRK_uh) == type(None): 
        midRK_uh = np.empty_like(u0h)
    if type(midRK_vh) == type(None): 
        midRK_vh = np.empty_like(v0h)

    for tt in t_computed_samples:
        ### RK4
        du0h_1, dv0h_1 = kol2D_obj.dynamics(u0h, v0h, du0h_1, dv0h_1) # k1
        midRK_uh[:, :] = u0h + 0.5*dt*du0h_1
        midRK_vh[:, :] = v0h + 0.5*dt*dv0h_1
            
        du0h_2, dv0h_2 = kol2D_obj.dynamics(midRK_uh, midRK_vh, du0h_2, dv0h_2) # k2
        midRK_uh[:, :] = u0h + 0.5*dt*du0h_2
        midRK_vh[:, :] = v0h + 0.5*dt*dv0h_2

        du0h_3, dv0h_3 = kol2D_obj.dynamics(midRK_uh, midRK_vh, du0h_3, dv0h_3) # k3
        midRK_uh[:, :] = u0h + dt*du0h_3
        midRK_vh[:, :] = v0h + dt*dv0h_3

        du0h_4, dv0h_4 = kol2D_obj.dynamics(midRK_uh, midRK_vh, du0h_4, dv0h_4) # k4
        u0h += dt * (du0h_1 + 2*du0h_2 + 2*du0h_3 + du0h_4) / 6
        v0h += dt * (dv0h_1 + 2*dv0h_2 + 2*dv0h_3 + dv0h_4) / 6

    if type(uh_reference) == type(None):
        uh_reference = np.zeros( (N_ref, N_ref), dtype=np.complex128 )
    if type(vh_reference) == type(None):
        vh_reference = np.zeros( (N_ref, N_ref), dtype=np.complex128 )

    uh_reference[:, :] = 0.0
    vh_reference[:, :] = 0.0
    
    uh_reference[ishift:ishift+2*N+1, ishift:ishift+2*N+1] = u0h
    vh_reference[ishift:ishift+2*N+1, ishift:ishift+2*N+1] = v0h
    
    all_data = np.empty(shape=(1, 2*N_ref*N_ref))
    all_data[0, 0:N_ref*N_ref] = np.reshape(
        np.fft.ifft2(np.fft.ifftshift(uh_reference)).real,
        (-1,)
    )
    all_data[0, N_ref*N_ref:] = np.reshape(
        np.fft.ifft2(np.fft.ifftshift(vh_reference)).real,
        (-1,)
    )
    
    res_dict = {'all_data':all_data}

    return res_dict
        












