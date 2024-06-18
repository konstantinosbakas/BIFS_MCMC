import numpy as np
import copy
from numpy.fft import fft2, ifft2, fftshift, ifftshift, rfft2, irfft2
import itertools 

class ParameterFunction:
    
    def __init__(self, image_shape, lamd, d):

        """
        Inputs:
            - dim (str): Dimensions of the k-space
            - lamd (float): lambda parameter of the parameter function (numerator)
            - d (float): d parameter of the parameter function (power of denominator)
            - image_shape (int): shape of the image
        """

        self.lamd = lamd
        self.d = d
        self.image_shape = image_shape
        self.dimensions = len(image_shape)
        
    def setDimensions(self):

        """
        Description: Set the dimensions of the image based on the shape
        """
        if self.dimensions == 2:    
            self.n1, self.n2 = self.image_shape
        else:
            self.n1, self.n2, self.n3 = self.image_shape
        
        if self.isOdd():
            self.dim_half = self.n1 // 2 + 1
        else:
            self.dim_half = self.n1 // 2 


    def kdist(self, n):
    
        """
            Create 1D centered k-space indexes 0:(n-1).
        
            Parameters --> n : int scalar size of array to be generated
            Returns --> kval : float numpy.ndarray array length n that gives 1D distance from wrapped origin at each index
        """
    
        kval = np.zeros(n, dtype=float)
    
        if (n % 2) == 0:
            kval[0:(n//2)] = -np.arange(n/2, 0, -1) 
            kval[(n//2):n] = np.arange(n/2)
        else:
            kval[0:(0 + n//2)] = -np.arange(np.floor(n/2), 0, -1)
            kval[(0 + n//2):n] = np.arange(np.ceil(n/2)) 
        
        return kval
        
    
    def origin_distances(self):
    
        """
            Generate matrix of distances from center of Fourier space 
            but shifted so origin is at index (0,0) of the matrix.
     
            Parameters -->  self.n1 : int scalar dimension of image in x-direction
                            self.n2 : int scalar dimension of image in y-direction
    
            Returns --> Xv : float numpy.ndarray
                            self.n1 x self.n2 array with distances from center of Fourier space, but
                            shifted so that origin is at index (0,0) of the array.
                            Leads to in-place version of np.sqrt(Xv ** 2 + Yv ** 2)
                            after np.meshgrid.
        """
        self.setDimensions()
        xvec = self.kdist(self.n1)
        yvec = self.kdist(self.n2)

        if self.dimensions == 2:
            Xv, Yv = np.meshgrid(xvec, yvec, indexing='ij')
            Xv **= 2
            Yv **= 2
            summ = Xv + Yv
        else:
            zvec = self.kdist(self.n3)
            Xv, Yv, Zv = np.meshgrid(xvec, yvec, zvec, indexing='ij')
            Xv **= 2
            Yv **= 2
            Zv **= 2
            summ = Xv + Yv + Zv
            
        radius = np.sqrt(summ)

        return radius

    def isOdd(self):
        if self.image_shape[0] % 2 != 0:
            return True
        return False
    
    def getOriginLocations(self, half_space):
        
        if self.isOdd():
            origin_coordinates = tuple(0 for _ in range(half_space.ndim))
        else:
            indx_pointer = itertools.product(['0',str(self.dim_half)], repeat=half_space.ndim) # pointer of combinations of indices
            indx_str = list(map(list, indx_pointer)) # list with combinations of indices (str)
            origin_coordinates = [tuple([int(float(j)) for j in i]) for i in indx_str] # list with combinations of indices (int)
        return origin_coordinates
    
    def getHalfParameterFunctionSpace(self, grid):

        N = self.image_shape[0]
        if self.isOdd():
            if self.dimensions == 2:
                out = grid[:, :(N // 2 + 1)]
            else:
                out = grid[:, :, :(N // 2 + 1)]
        else:
            if self.dimensions == 2:
                out = grid[:, :(N // 2 + 1)]
            else:
                out = grid[:, :, :(N // 2 + 1)]
        return out

    def setDistances(self):
        """
        Description: Sets the distance matrix and extract the half of it.
                     Taking advantage of the Hermitian property of the Fourier space.
        """
        self.setDimensions()
        distances = self.origin_distances()
        distances_shifted = fftshift(distances)
        parameter_function_half_space = self.getHalfParameterFunctionSpace(distances_shifted)
        origin_locations = self.getOriginLocations(parameter_function_half_space)
        for coord in origin_locations:
            parameter_function_half_space[coord] = 0.5
        return parameter_function_half_space

    def get(self):

        """
            Parameter function for the prior:
            λ -> constant for mean
            c -> constant of proportionality for standard deviation
        """
        distances = self.setDistances().flatten()
        means = self.lamd / distances**self.d
        sds = means
        return [means, sds]
    