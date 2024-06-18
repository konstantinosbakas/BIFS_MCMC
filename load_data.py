import numpy as np
from scipy.ndimage import gaussian_filter

class LoadData:
    
    def __init__(self,
                 dimensions,
                 slice, 
                 axis,
                 type_noise, 
                 sd_noise,
                 isTumor,
                 centerTumor,
                 radiusTumor,
                 intensity,
                 seed):
        
        # Input  
        self.dimensions = dimensions
        self.slice = slice
        self.axis = axis
        self.type_noise = type_noise
        self.sd_noise = sd_noise
        self.isTumor = isTumor
        self.centerTumor = centerTumor
        self.radiusTumor = radiusTumor
        self.intensity = intensity
        self.seed = seed
        
        self.normalizedImage = None
        self.NoisyImage = None
        
        # Output
        self.suboptimal = None
        self.true = None
        self.gray_matter_locations = None
        self.white_matter_locations = None
        self.csf_locations = None

    def selectImage2D(self, image):
        if self.axis == 0:
            image2D = image[self.slice, :, :]
        elif self.axis == 1:
            image2D = image[:, self.slice, :]
        elif self.axis == 2:
            image2D = image[:, :, self.slice]
        else:
            print("Error: select axis {0, 1, 2}")
        self.true = image2D
    
    def selectDimensions(self, image):
        if self.dimensions == 2:
            self.selectImage2D(image)
        elif self.dimensions == 3:
            self.true = image
        else:
            print("Error: select dimensions {2,3}")

    def readImage(self):
        print("\nLoading Image...")
        Adim = 128  # Structural MRI dimension
        Zdim = 128  # Number of structural MRI slices

        gray_intensity = 2
        white_intensity = 1
        csf_intensity = 0
        noise_SD = 0.2
        
        gmask = np.loadtxt("Data/gm_2mm_3D_mask.txt", delimiter="\t")
        wmask = np.loadtxt("Data/wm_2mm_3D_mask.txt", delimiter="\t")
        cmask = np.loadtxt("Data/csf_2mm_3D_mask.txt", delimiter="\t")

        AdimHalf = Adim // 2
        AdimMinone = Adim - 1
        AdHminone = AdimHalf - 1

        xA = yA = np.arange(-AdimHalf, AdHminone + 1)

        Asize = Adim**2
        ZAsize = Asize * Zdim

        wmsk = np.round(wmask[:, 3]).astype(int)
        gmsk = np.round(gmask[:, 3]).astype(int)
        cmsk = np.round(cmask[:, 3]).astype(int)

        wmsk = wmsk.reshape((Adim, Adim, Zdim), order='F')
        gmsk = gmsk.reshape((Adim, Adim, Zdim), order='F')
        cmsk = cmsk.reshape((Adim, Adim, Zdim), order='F')
        
        self.white_matter_locations = np.where(wmsk != 0, 1, 0)
        self.gray_matter_locations = np.where(gmsk != 0, 1, 0)
        csf_locations = np.where(cmsk != 0, 1, 0)
        brain_msk = self.white_matter_locations + self.gray_matter_locations + csf_locations
        out_locations = np.where(brain_msk == 0, 1, 0)
        self.csf_out_locations = csf_locations + out_locations
        
        wmsk2 = np.ravel(np.transpose(wmsk, (2, 1, 0)))
        gmsk2 = np.ravel(np.transpose(gmsk, (2, 1, 0)))
        cmsk2 = np.ravel(np.transpose(cmsk, (2, 1, 0)))

        with open("Data/wmask3D.dat", "wb") as wmas:
            wmsk2.tofile(wmas)
        with open("Data/gmask3D.dat", "wb") as gmas:
            gmsk2.tofile(gmas)
        with open("Data/cmask3D.dat", "wb") as cmas:
            cmsk2.tofile(cmas)

        Map3D = gray_intensity * gmsk + white_intensity * wmsk + csf_intensity * cmsk
        self.selectDimensions(Map3D)

    def normalize(self):
        normalized = (self.true.flatten()-min(self.true.flatten()))/(max(self.true.flatten())-min(self.true.flatten()))
        self.normalizedImage = np.reshape(normalized, np.shape(self.true))
        self.true = self.normalizedImage
    
    def addTumor2D(self):
        n1, n2 = np.shape(self.true)
        grid = (n1, n2)
        hotspot = np.zeros(grid)
                
        for i in range(n1):
            for j in range(n2):
                if np.sqrt( (i-self.centerTumor[0])**2 + (j-self.centerTumor[1])**2 ) < self.radiusTumor:
                    hotspot[i, j] = 1
    
        # This keeps it in gray matter only — you could have it only in white matter
        hotspot = self.intensity * hotspot
        
        # This next part just smooths it a bit
        trim = range(1, n2-1)
        lower = range(n2-2)
        upper = range(2, n2)
        hotspot[trim,trim] = 0.2*(hotspot[trim, trim] + hotspot[lower, trim] + hotspot[upper, trim] + hotspot[trim, lower] + hotspot[trim, upper])                      
        
        # Add the tumot into the data (image)
        self.normalizedImage = self.normalizedImage + hotspot
    
    def addTumor3D(self):
        n1, n2, n3 = np.shape(self.true)
        grid = (n1, n2, n3)
        hotspot = np.zeros(grid)
                
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    if np.sqrt( (i-self.centerTumor[0])**2 + (j-self.centerTumor[1])**2 + (k-self.centerTumor[2])**2) < self.radiusTumor:
                        hotspot[i, j, k] = 1
    
        # This keeps it in gray matter only — you could have it only in white matter
        hotspot = self.intensity * hotspot
        
        # This next part just smooths it a bit
        trim = range(1, n2-1)
        lower = range(n2-2)
        upper = range(2, n2)
        hotspot[trim,trim, trim] = 0.2*(hotspot[trim, trim, trim] + hotspot[lower, trim, trim] + hotspot[upper, trim, trim] + hotspot[trim, upper, trim] + hotspot[trim, lower, trim] + hotspot[trim, trim, lower] + hotspot[trim, trim, upper])                      
        
        # Add the tumot into the data (image)
        self.normalizedImage = self.normalizedImage + hotspot

    def addNoise(self):
        np.random.seed(self.seed)
        image_shape = np.shape(self.normalizedImage)
        if self.type_noise == "gaussian":
            noise = np.random.normal(0, self.sd_noise, image_shape)
            noisy_image = self.normalizedImage + noise
        elif self.type_noise == "shot":
            noisy_image = np.random.poisson(self.normalizedImage * self.sd_noise) / self.sd_noise
        elif self.type_noise == "uniform":
            noise = np.random.uniform(0, self.sd_noise, image_shape)
            noisy_image = self.normalizedImage + noise
        elif self.type_noise == "correlated":
            noise = np.random.normal(0, self.sd_noise, image_shape)
            noisy_image = self.normalizedImage + gaussian_filter(noise, 2)
        else:
            print("There is no such type of noise")
        
        self.suboptimal = noisy_image
    
    def preprocess(self):
        self.readImage() # Read image
        self.normalize() # Normalize image
        if self.isTumor:
            if self.dimensions == 2:
                self.addTumor2D()
            else: 
                self.addTumor3D()
        self.addNoise() # Add noise
        
    def getTrueImage(self):
        return self.true

    def getSuboptimalImage(self):
        self.preprocess()
        return self.suboptimal
    
    def getWhiteMatter(self):
        if self.axis == 0:
            white_matter = self.white_matter_locations[self.slice, :, :]
        elif self.axis == 1:
            white_matter = self.white_matter_locations[:, self.slice, :]
        elif self.axis == 2:
            white_matter = self.white_matter_locations[:, :, self.slice]
        else:
            print("Error: select axis {0, 1, 2}")
        
        return white_matter
    
    def getGrayMatter(self):
        if self.axis == 0:
            gray_matter = self.gray_matter_locations[self.slice, :, :]
        elif self.axis == 1:
            gray_matter = self.gray_matter_locations[:, self.slice, :]
        elif self.axis == 2:
            gray_matter = self.gray_matter_locations[:, :, self.slice]
        else:
            print("Error: select axis {0, 1, 2}")
        
        return gray_matter

    def getCSFOut(self):
        if self.axis == 0:
            csf_out = self.csf_out_locations[self.slice, :, :]
        elif self.axis == 1:
            csf_out = self.csf_out_locations[:, self.slice, :]
        elif self.axis == 2:
            csf_out = self.csf_out_locations[:, :, self.slice]
        else:
            print("Error: select axis {0, 1, 2}")
        
        return csf_out
    