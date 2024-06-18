import numpy as np

class Noise:
    
    def __init__(self, data, sd, noise_type, seed):
        self.data = data
        self.sd = sd
        self.noise_type = noise_type
        self.seed = seed
        
        self.get_data_shape = np.shape(self.data)
        self.n1, self.n2 = self.get_data_shape
        
        
    def get_shape(self):
        return np.shape(self.data)
    
    def get(self):
        return self.data

    def add_noise(self):
        np.random.seed(self.seed)
        
        if self.noise_type == "gaussian":
            
            mean = 0
            noise = np.random.normal(mean, self.sd, self.get_shape())

        elif self.noise_type == "shot":
            noise = np.random.poisson(self.sd, self.get_shape()) / self.sd

        elif self.noise_type == "uniform":
            
            noise = np.random.uniform(0, self.sd, self.get_shape())
            
        elif self.noise_type == "poisson":
            
            myimage = self.sd * np.ones(self.get_shape())
            noise = np.random.poisson(lam = myimage, size = self.get_shape())

        noisy_data = self.data + noise
        self.data = noisy_data