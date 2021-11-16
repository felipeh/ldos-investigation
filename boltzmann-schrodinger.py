import pyfftw
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import scipy.signal
#import seaborn as sns
from torus_handler import TorusHandler

def l2_prod(f, g):
    return np.sum(f * g.conj()) * (2*np.pi/f.shape[0])**2

def eval_wavepacket(X,Y,x0,xi0,r):
    phase = np.exp(1j*(xi0[0]*X+xi0[1]*Y))
    envelope = np.exp(-((X-x0[0])**2+(Y-x0[1])**2)/r**2)
    return (1./r) * phase * envelope

def propagator(t,kx,ky):
    return np.exp(-1j * t * (kx**2+ky**2) / 2)

def rearrange_ft(psihat):
    N = psihat.shape[0]
    half_N = int(N/2)
    rearranged = np.zeros_like(psihat)
    rearranged[0:half_N,0:half_N] = psihat[half_N:,half_N:]
    rearranged[0:half_N,half_N:] = psihat[half_N:,0:half_N]
    rearranged[half_N:,0:half_N] = psihat[0:half_N,half_N:]
    rearranged[half_N:,half_N:] = psihat[0:half_N,0:half_N]
    return rearranged

def iterate_schrodinger(psi0, V, dt, ksteps, torus, multiplier=None,
                        compute_norms=False):
    psi = torus.alloc_aligned_memory()
    psi_hat = torus.alloc_aligned_memory()

    psi[:] = psi0
    # do a first half-step of collision against potential
    collider = torus.alloc_aligned_memory()
    collider[:] = np.exp(1j * V * dt)
    propagator = torus.alloc_aligned_memory()
    propagator[:] = np.exp(-1j*dt*(torus.kx**2+torus.ky**2)/2)
    psi[:] *= np.exp(1j * V * dt / 2)

    norms = np.zeros(ksteps)
    psi_free_hat = torus.alloc_aligned_memory()
    torus.fft(psi0, psi_free_hat)
    psinorm = 1

    for knt in range(ksteps):
        torus.fft(psi, psi_hat)
        if knt == 0:
            psinorm = np.linalg.norm(psi_hat)
        psi_hat[:] *= propagator
        psi_free_hat[:] *= propagator
        torus.ifft(psi_hat, psi)
        psi[:] *= collider
        if compute_norms:
            norms[knt] = np.linalg.norm(psi_free_hat - psi_hat)

    psi[:] /= np.exp(1j * V * dt / 2)
    if compute_norms:
        return psi, norms / psinorm
    return psi

def compute_ldos(V, E, delta_E, delta_E_fourier, dt, torus):
    psi = torus.alloc_aligned_memory()
    psi_hat = torus.alloc_aligned_memory()

    # first start with initial condition supported on Fourier annulus
    E_fourier = (torus.kx**2 + torus.ky**2)/2
    # in principle one should take delta_E_fourier=infty, but i think
    # it's fine to take it just a bit larger than delta_E
    psi_hat[:] = np.exp(-((E_fourier-E)/delta_E_fourier)**2)
    psi_hat[:] *= np.exp(1j * np.pi * (torus.kx + torus.ky))
    torus.ifft(psi_hat, psi)

    # we are going to integrate evolution of psi and store it in LDOS
    LDOS = torus.alloc_aligned_memory()
    LDOS = 0*LDOS
    # we also need the backwards time evolution
    psi_backwards = torus.alloc_aligned_memory()
    psi_back_hat = torus.alloc_aligned_memory()
    psi_backwards[:] = psi
    psi_back_hat[:] = psi_hat

    proj_hat = lambda s: np.exp(-(delta_E)**2 * s**2) *\
                            np.exp(1j * (E * s)) * delta_E

    LDOS += dt * psi * proj_hat(0)

    # the time horizon goes as 1/delta_E
    T = 2.5/delta_E
    t = 0

    # needed for time evolution
    collider = torus.alloc_aligned_memory()
    collider[:] = np.exp(1j * V * dt/2)
    propagator = torus.alloc_aligned_memory()
    propagator[:] = np.exp(-1j*dt*(torus.kx**2+torus.ky**2)/2)

    print(T)
    while t < T:
        psi *= collider
        psi_backwards /= collider
        torus.fft(psi, psi_hat)
        torus.fft(psi_backwards, psi_back_hat)
        psi_hat[:] *= propagator
        psi_back_hat[:] /= propagator
        torus.ifft(psi_hat, psi)
        torus.ifft(psi_back_hat, psi_backwards)
        psi[:] *= collider
        psi_backwards[:] /= collider
        t += dt
        print(t, proj_hat(t).real)

        LDOS += dt * (psi * proj_hat(t) + psi_backwards * proj_hat(-t))
    return LDOS, psi


def approximate_projection(psi0, xi0, x0, rxi, rx, torus):
    psi = torus.alloc_aligned_memory()
    psi_hat = torus.alloc_aligned_memory()
    psi[:] = psi0
    torus.fft(psi, psi_hat)
    psi_hat *= np.exp(-((torus.kx-xi0[0])**2+(torus.ky-xi0[1])**2)/rxi**2/4)
    torus.ifft(psi_hat, psi)
    psi *= np.exp(-((torus.x-x0[0])**2+(torus.y-x0[1])**2)/rx**2/4)
    return psi

def random_lattice(dlattice, torus):
    l_shape = int(torus.N / dlattice)
    return np.random.randn(l_shape,l_shape)

def random_V(rho, dlattice, torus, seed_lattice=None):
    kernel = torus.alloc_aligned_memory()
    noise_lattice = torus.alloc_aligned_memory()
    noise_lattice[:] = 0
    l_shape = noise_lattice[::dlattice,::dlattice].shape
    if seed_lattice is not None:
        noise_lattice[::dlattice,::dlattice] = seed_lattice
    else:
        noise_lattice[::dlattice,::dlattice] = np.random.randn(*l_shape)

    kernel[:] = np.exp(-((torus.x-np.pi)**2+(torus.y-np.pi)**2)/rho**2/2)
    V = torus.conv_fft(noise_lattice, kernel)
    V.imag[:] = 0
    return V

def gaussian_field(rho, torus):
    noise = np.random.randn(torus.N, torus.N)+1j*np.random.randn(torus.N,torus.N)
    noise *= np.exp(-(torus.kx**2+torus.ky**2)/rho**2)
    return torus.ifft_copy(noise).real + 0j

def localization_landscape(V,torus):
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg as conjugate_gradient

    u = torus.alloc_aligned_memory()
    u_hat = torus.alloc_aligned_memory()

    symbol = torus.kx**2 + torus.ky**2

    symbol_inv  = symbol
    symbol_inv[symbol_inv < 1e-8] = 1e-8
    symbol_inv = 1./symbol_inv

    def apply_H(x, v, v_hat, N):
        v[:] = x.reshape(N,N)
        torus.fft(v, v_hat)
        v_hat *= symbol
        torus.ifft(v_hat, v)
        return v.reshape(-1) + V.reshape(-1)*x

    def apply_Linv(x, v, v_hat, N):
        v[:] = x.reshape(N,N)
        torus.fft(v, v_hat)
        v_hat *= symbol_inv
        torus.ifft(v_hat, v)
        return v.reshape(-1)

    apH = lambda x: apply_H(x, u, u_hat, torus.N)
    apL = lambda x: apply_Linv(x, u, u_hat, torus.N)

    Nsqrd = torus.N * torus.N
    H = LinearOperator((Nsqrd,Nsqrd), apH)
    Linv = LinearOperator((Nsqrd,Nsqrd), apL)

    all_ones = np.ones(Nsqrd)
    x, info =  conjugate_gradient(H, all_ones, M=Linv)
    if info != 0:
        print("something went wrong when computing the landscape")
    return x.reshape(torus.N,torus.N)

def main():
    eps = 0.07
    kappa = 0.5
    N = 1024
    torus = TorusHandler(N, cache=".fftw_cache")
    print(np.max(torus.kx))

    x0 = np.array((0.1,np.pi))
    xi0 = np.array((1./eps**(2+kappa/2),0))

    randomness = np.random.rand(N//2,N//2)
    V = random_V(5./N, 2, torus, seed_lattice=randomness) * 5000
    #V = random_V(5./N, 4, torus) * 5000
    print(np.max(V))

    print("computing localization landscape..")
    u = localization_landscape(V, torus)

    print("now computing ldos")

    speed = 100
    ldos,psi = compute_ldos(V, speed**2/2,
                            speed/10, 5*speed, 0.01/speed, torus)

    plt.figure()
    plt.imshow(V.real)
    plt.figure()
    plt.imshow(ldos.real**2 + ldos.imag**2)
    plt.figure()
    plt.imshow(u.real)

    #plt.subplot(2,2,1)
    #plt.imshow(psi.real)
    np.save('ldos.npy', ldos)
    np.save('psi.npy', psi)
    np.save('LL.npy', u)
    plt.show()

def test_localization_landscape():
    N = 1024
    torus = TorusHandler(N, cache=".fftw_cache")

    randomness = np.random.rand(N//2,N//2)
    V = random_V(5./N, 2, torus, seed_lattice=randomness) * 500
    print(np.max(V))

    u = localization_landscape(V, torus)

    plt.figure()
    plt.imshow(V.real)
    plt.figure()
    plt.imshow(u.real)
    plt.show()


if __name__ == "__main__":
    main()
    #test_localization_landscape()


