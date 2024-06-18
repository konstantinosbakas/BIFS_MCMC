import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift, rfft2, irfft2, fftn, ifftn, rfftn, irfftn
import itertools 

class Fourier3D:
    
    def __init__(self, data):
        self.data = data
        self.fourier = None
        self.N = None
        self.d = None
        self.full_space_shape = None
        
    def setParameters(self):
        self.N = self.fourier.shape[0]
        if self.isOdd():
            self.d = self.N // 2 + 1
        else:
            self.d = self.N // 2 
        
        self.full_space_shape = self.data.shape     
    
    def transform(self):
        """
            Transform data (FFT): Image Space -> Fourier Space
        """       
        self.fourier = rfftn(self.data, norm="ortho")
        self.setParameters()
        return self.fourier

    def isOdd(self):
        if self.fourier.shape[0] % 2 != 0:
            return True
        return False
     
    def getOriginLocations(self):

        if self.isOdd():
            origin_coordinates = tuple(0 for _ in range(self.data.ndim))
        else:
            indx_pointer = itertools.product(['0',str(self.d)], repeat=self.data.ndim) # pointer of combinations of indices
            indx_str = list(map(list, indx_pointer)) # list with combinations of indices (str)
            origin_coordinates = [tuple([int(float(j)) for j in i]) for i in indx_str] # list with combinations of indices (int)
        return origin_coordinates
    
    def getOriginValues(self, locations):
        if self.isOdd():
            origin_values = self.fourier[locations]
        else:
            origin_values = [self.fourier[coord] for coord in locations] 
        return origin_values
    
    def inverse(self, matrix):
        """
            Transform data (FFT): Fourier Space -> Image Space
        """

        out = irfftn(matrix, norm="ortho")
        return out
        
    def estimateVariance(self):
        """
            Estimate variance in Fourier Space
        """
 
        varEstimateImageSpace = np.var(self.data[31:35, 29:33]) 
        return varEstimateImageSpace / 2

        