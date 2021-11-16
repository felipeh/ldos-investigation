import pyfftw
import numpy as np

def my_fft2(psi, h):
    return np.fft.fft2(psi) * h**2 / (2*np.pi)

def my_ifft2(psihat, h):
    return np.fft.ifft2(psihat) * (2*np.pi) / h**2

def my_fftfreq(N,h):
    return np.fft.fftfreq(N,h) * (2*np.pi)

def rearrange_ft(psihat):
    N = psihat.shape[0]
    rearranged = np.zeros_like(psihat)
    rearranged[0:N/2,0:N/2] = psihat[N/2:,N/2:]
    rearranged[0:N/2,N/2:] = psihat[N/2:,0:N/2]
    rearranged[N/2:,0:N/2] = psihat[0:N/2,N/2:]
    rearranged[N/2:,N/2:] = psihat[0:N/2,0:N/2]
    return rearranged

_unique_separator = b'IBETTHISSTRINGISNTINTHEWISDOM'
def read_wisdom(stream):
    wisdom = tuple(stream.read().split(_unique_separator))
    assert len(wisdom) == 3
    return wisdom

def write_wisdom(wisdom, f):
    outf = open(f,'wb')
    outf.write(_unique_separator.join(wisdom))
    outf.close()

class TorusHandler:
    '''Wraps FFTW and scales frequencies for our calculations on [0,2pi]^2'''
    def __init__(self, N, cache=None):
        self.N = N
        h = 2*np.pi / N
        self.h = h
        x = h * np.array(range(N))
        self.X, self.Y = np.meshgrid(x,x)
        self.x = self.X
        self.y = self.Y

        k = np.fft.fftfreq(N,h) * (2*np.pi)
        self.kx,self.ky = np.meshgrid(k,k)

        self.a = pyfftw.empty_aligned((N,N), dtype='complex128')
        self.b = pyfftw.empty_aligned((N,N), dtype='complex128')
        self.c = pyfftw.empty_aligned((N,N), dtype='complex128')
        self.d = pyfftw.empty_aligned((N,N), dtype='complex128')


        opt_flag = "FFTW_MEASURE"
        if cache is not None:
            try:
                wisdom_file = open(cache, 'rb')
                cached_wisdom = read_wisdom(wisdom_file)
                pyfftw.import_wisdom(cached_wisdom)
                wisdom_file.close()
            except IOError:
                # no wisdom here!
                print("No wisdom found -- optimizing FFTW with PATIENT flag")

            opt_flag = "FFTW_PATIENT"

        print("Optimizing ... ")
        self.forward = pyfftw.FFTW(self.a, self.b,
                                   axes=(0,1),
                                   threads=4,
                                   flags=(opt_flag,))
        self.inverse = pyfftw.FFTW(self.c, self.d,
                                   direction='FFTW_BACKWARD',
                                   axes=(0,1),
                                   threads=4,
                                   flags=(opt_flag,))
        print("Done!")
        if cache is not None:
            wisdom = pyfftw.export_wisdom()
            write_wisdom(wisdom, cache)

        self.forward_assigned = True
        self.backward_assigned = True

    def alloc_aligned_memory(self):
        return pyfftw.empty_aligned((self.N,self.N), dtype='complex128')

    def write_and_fft_and_write(self, psi, psihat):
        self.a[:] = psi
        self.forward()
        psihat[:] = self.b

    def write_and_ifft_and_write(self, psihat, psi):
        self.c[:] = psihat
        self.inverse()
        psi[:] = self.d

    def fft_copy(self, psi):
        '''Convenience FFT that takes any (not aligned) input and output, at
        the cost of copying input into aligned array.'''
        if not self.forward_assigned:
            self.forward.update_arrays(self.a, self.b)
            self.forward_assigned = True

        self.a[:] = psi

        self.forward()
        return np.copy(self.b)

    def ifft_copy(self, psi):
        '''Convenience IFFT that takes any (not aligned) input and output, at
        the cost of copying input into aligned array.'''
        if not self.backward_assigned:
            self.inverse.update_arrays(self.c, self.d)
            self.backward_assigned = True
        self.c[:] = psi
        self.inverse()
        return np.copy(self.d)


    def fft(self, psi, psihat):
        '''Optimal use of FFT, no copying of any memory involved'''
        # here psi and psihat should be constructed with alloc_aligned_memory
        self.forward_assigned = False
        self.forward.update_arrays(psi, psihat)
        self.forward()

    def ifft(self, psihat, psi):
        # same caveat as fft
        self.backward_assigned = False
        self.inverse.update_arrays(psihat, psi)
        self.inverse()

    def conv_fft(self, f1, f2):
        # a and b contain the ffts of f1 and f2
        self.forward.update_arrays(f1, self.a)
        self.forward()
        self.forward.update_arrays(f2, self.b)
        self.forward()
        # multiply the fft values in c
        self.c[:] = self.a * self.b
        # take the inverse transform
        self.inverse.update_arrays(self.c,self.d)
        self.inverse()
        # so that it's back to normal
        self.forward.update_arrays(self.a,self.b)
        return np.copy(self.d)
