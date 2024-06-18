import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift, rfft2, irfft2


class Fourier:
    
    def __init__(self, data, method, figure, trim):
        self.data = data
        self.method = method
        self.figure = figure
        self.trim = trim

    def transform(self):
        """
            Transform data (FFT): Image Space -> Fourier Space
        """
        if self.method == "real":
            fourier = rfft2(self.data, norm="ortho")
            # fourier = rfft2(self.data)
        elif self.method == "complex":
            fourier = fftshift(fft2(self.data))
        if self.figure:
            fourier = fftshift(np.log(abs(fft2(self.data)))) 
        return fourier

### GET_POSITIONS FUNCTION (SEE TEST3.PY)
    
    def inverse(self, matrix):
        """
            Transform data (FFT): Fourier Space -> Image Space
        """
        if self.method == "real":
            out = irfft2(matrix, s=self.data.shape, norm="ortho")
            # out = irfft2(matrix, s=self.data.shape)
        elif self.method == "complex":
            out = ifft2(ifftshift(matrix))
        return out
        
    def estimateVariance(self):
        """
            Estimate variance in Fourier Space
        """
        
        # Patch of pixels of simulated image to estimate the variance of the noise
        # varEstimateImageSpace = np.var(matrix[0:30, 0:30])
        
        # Patch of pixels of real image (UCSF) to estimate the variance of the noise
        # varEstimateImageSpace = np.var(matrix[0:50, 110:230])
        # varEstimateImageSpace = np.var(matrix[:, 232:256])

        # Patch of pixels of real image (ASL perfusion) to estimate the variance of the noise
        # varEstimateImageSpace = np.var(self.data[60:70, 60:70]) 
        # varEstimateImageSpace = np.var(self.data[36:40, 29:33]) 
        # varEstimateImageSpace = np.var(self.data[0:10, 0:10, 0:10]) 
        
        varEstimateImageSpace = np.var(self.data[:self.trim, :self.trim])
        
        return varEstimateImageSpace / 2
    
    def estimatePrecision(self, matrix):
        """
            Estimate precision in Fourier Space
        """
        varEstimateImageSpace = np.var(matrix[0:30, 0:30])
        precisionEstimateFourierSpace = 2 / varEstimateImageSpace
        
        return precisionEstimateFourierSpace
    
        