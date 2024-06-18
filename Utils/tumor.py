import copy
import numpy as np

class Tumor:
    
    def __init__(self, data, center, radius, intensity):
        self.data = data
        self.center = center
        self.radius = radius
        self.intensity = intensity
    
    def get_data2D_with_tumor(self):
        return self.data
    
    def get_data3D_with_tumor(self):
        return self.data
    
    def add_3D(self):
        """
             Add a tumor (hotspot) as circle into the data (brain): 
            Shape characteristics (center, radius): Center and radius of the tumor
            Intensity and matter has to been defined as well
        """
        n1, n2, n3 = np.shape(self.data)
        grid = (n1, n2, n3)
        hotspot = np.zeros(grid)

        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    if np.sqrt((i-self.center)**2 + (j-self.center)**2 + (k-self.center)**2) < self.radius:
                        hotspot[i, j, k] = 1
                        
        # This keeps it in gray matter only — you could have it only in white matter
        hotspot = self.intensity * hotspot                    
    
        # Add the tumot into the data (image)
        copy_data= copy.deepcopy(self.data)
        data_tumor = copy_data + hotspot
        self.data = data_tumor
        
    def add_2D(self):
        """
             Add a tumor (hotspot) as circle into the data (brain): 
            Shape characteristics (center, radius): Center and radius of the tumor
            Intensity and matter has to been defined as well
        """

        n1, n2 = np.shape(self.data)
        grid = (n1, n2)
        hotspot = np.zeros(grid)
                
        for i in range(n1):
            for j in range(n2):
                if np.sqrt( (i-self.center[0])**2 + (j-self.center[1])**2 ) < self.radius:
                    hotspot[i, j] = 1
    
        # This keeps it in gray matter only — you could have it only in white matter
        hotspot = self.intensity * hotspot
        
        # This next part just smooths it a bit
        trim = range(1, n2-1)
        lower = range(n2-2)
        upper = range(2, n2)
        hotspot[trim,trim] = 0.2*(hotspot[trim, trim] + hotspot[lower, trim] + hotspot[upper, trim] + hotspot[trim, lower] + hotspot[trim, upper])                      
        
        # Add the tumot into the data (image)
        copy_data= copy.deepcopy(self.data)
        data_tumor = copy_data + hotspot
        self.data = data_tumor
        