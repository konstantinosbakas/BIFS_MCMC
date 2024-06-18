from parameter_function import ParameterFunction
from adaptive import Adaptive
from Utils.fourier import Fourier
from Utils.MAP import MAP
import Utils.tools as F
import numpy as np

class BIFS:

    def __init__(self,
                 image,  # Input image
                 d, lamd,  # Parameter function for (ρ, θ)
                 trim, # Value to trim the image and estimate the standard deviation of noise
                 sample,  # Number of samples
                 burn,  # Number of burn samples
                 thin,  # Thinning
                 batches,  # Number of batches (for adaptation)
                 batch_size,  # Iterations for each bach (for adaptation)
                 sigma_q,  # Variances for the proposal distribution
                 adapting,  # Adaptive MCMC option
                 type_estimation, # Type of method (mean, median, quantile, MAP)
                 q_percentile, # q-percentile for type_estimation = quantile (manual)
                 seed):

        self.valuesMAP = None
        self.num_signals = None
        self.image_est = None
        self.image_map_est = None
        self.theta_chains = None
        self.rho_chains = None
        self.a_chains = None
        self.b_chains = None
        self.signal_chains = None
        self.image_chains = None
        self.sigma_q = None
        self.r = None
        self.psi = None
        self.sigma2 = None
        self.mu_rho = None
        self.sigma_rho2 = None
        self.initial_values = None
        self.acceptanceRatios = None
        self.rho_MAP_est = None 
        self.theta_MAP_est = None
        
        self.half_shape = None

        # Initiate data for two images
        self.image = image

        # Initiate parameters for parameter function (Prior)
        self.d = d
        self.lamd = lamd
    
        self.trim = trim
    
        # Initiate MCMC configurations
        self.sample = sample
        self.burn = burn
        self.thin = thin
        self.adapting = adapting
        self.batches = batches
        self.batch_size = batch_size
        self.seed = seed
        self.execution_time = 0
        self.q_percentile = q_percentile
        
        # Initiate dimensions and origin index
        self.n1, self.n2 = np.shape(self.image)
        self.pixels = self.n1 * self.n2
        self.sigma_q = sigma_q

        self.type_estimation = type_estimation

        self.sigma_q = np.full(self.num_signals, self.sigma_q) # Initialize proposed variances
        self.acceptanceRatios = np.zeros(self.num_signals) # Initialize acceptance probabilities
        
        self.origin = None  

        if self.n1 % 2 == 0 and self.n2 % 2 == 0:
            self.isEven = True
        self.isEven = False
              
    
    def toFourierSpace(self):
        FFT = Fourier(data=self.image, method='real', figure=False, trim=self.trim)
        self.signal = FFT.transform()
        self.half_shape = self.signal.shape
        self.num_signals = self.half_shape[0] * self.half_shape[1]
        if self.isEven:
            self.origin = [0, self.n1/2]
        else:
            self.origin = 0

    def toImageSpace(self, signal):
        FFT = Fourier(data=self.image, method='real', figure=False, trim=self.trim)
        out = FFT.inverse(signal)
        return out
    
    def estimateLikelihoodVariance(self):
        FFT = Fourier(data=self.image, method='real', figure=False, trim=self.trim)
        sigma2 = FFT.estimateVariance() 
        return sigma2

    def setLikelihoodParameters(self):
        self.r = np.absolute(self.signal).flatten()
        self.psi = np.angle(self.signal).flatten()
        sigma2 = self.estimateLikelihoodVariance()
        self.sigma2 = np.full(self.num_signals, sigma2)

    def setPriorParameters(self):
        paramFuncRho = ParameterFunction(np.shape(self.image), self.lamd, self.d)
        mu_rho, sigma_rho = paramFuncRho.get()
        self.mu_rho = mu_rho
        self.sigma_rho2 = sigma_rho**2    

    def setInitialValues(self):
        rho = MAP(self.r, self.sigma2, self.mu_rho, self.sigma_rho2, self.num_signals)
        rho_init = rho.calculate_MAP()
        theta_init = self.psi
        self.initial_values = [rho_init, theta_init]

    def setChains(self):
        rho_chains = np.zeros((self.sample, self.num_signals))
        theta_chains = np.zeros((self.sample, self.num_signals))
        rho_chains[0, :] = self.initial_values[0]
        theta_chains[0, :] = self.initial_values[1]
        return [rho_chains, theta_chains]

    def saveChains(self, rho_chains, theta_chains):
        self.rho_chains = rho_chains
        self.theta_chains = theta_chains

    def hermitianSymmetric(self, vector):
        origin = np.real(vector[self.origin])
        first_half = vector[:self.origin]
        second_half = np.conj(first_half[::-1])
        out_flat = np.append(np.append(first_half, origin), second_half)
        out_reshape = np.reshape(out_flat, (self.n1, self.n2))
        return out_reshape

    def saveOriginValues(self, sig):
        if self.isEven:
            sig[0, 0] = self.signal[0, 0]
            sig[0, self.n2/2] = self.signal[0, self.n2/2]
            sig[self.n1/2, 0] = self.signal[self.n1/2, 0]
            sig[self.n1/2, self.n2/2] = self.signal[self.n1/2, self.n2/2]
        else:
            sig[0,0] = self.signal[0,0]
        return sig

    def getMapEstimates(self):
        self.rho_MAP_est, self.theta_MAP_est = self.initial_values
        a_est, b_est = F.polarToCartesian(self.rho_MAP_est, self.theta_MAP_est) # Get estimates in Cartesian coordinates
        half_signal_est = np.vectorize(complex)(a_est, b_est) # Get complex estimates
        half_signal_est_reshaped = np.reshape(half_signal_est, self.half_shape)
        half_signal_est_reshaped = self.saveOriginValues(half_signal_est_reshaped) # Keep observed origin values
        self.image_map_est = self.toImageSpace(half_signal_est_reshaped) # Get estimates in image space
        return self.image_map_est

    def polarToCartesianChains(self):
        self.a_chains, self.b_chains = F.polarToCartesian(self.rho_chains, self.theta_chains)

    def toComplexChains(self):
        self.signal_chains = np.vectorize(complex)(self.a_chains, self.b_chains)
        self.signal_chains = np.reshape(self.signal_chains, (self.sample, self.half_shape[0], self.half_shape[1]))
        
    def saveOriginValuesChains(self):
        if self.isEven:
            self.signal_chains[:, 0, 0] = self.signal[0, 0]
            self.signal_chains[:, 0, self.n2/2] = self.signal[0, self.n2/2]
            self.signal_chains[:, self.n1/2, 0] = self.signal[self.n1/2, 0]
            self.signal_chains[:, self.n1/2, self.n2/2] = self.signal[self.n1/2, self.n2/2]
        else:
            # self.signal_chains[:, self.origin] = self.signal[0, 0]
            self.signal_chains[:, 0, 0] = self.signal[0, 0]
            
    def toImageSpaceChains(self):
        self.image_chains = np.zeros((self.sample, self.n1, self.n2))
        for i in range(self.sample):
            sample_reshaped = np.reshape(self.signal_chains[i,:], self.half_shape)
            self.image_chains[i,:,:] = self.toImageSpace(sample_reshaped)
    
    def setImagePosteriorEstimates(self):  
        if self.type_estimation == "mean":
            self.image_est = np.mean(self.image_chains, axis=0)
        elif self.type_estimation == "median":
            self.image_est = np.median(self.image_chains, axis=0)
        elif self.type_estimation == "quantiles":
            self.image_est = np.percentile(self.image_chains, self.q_percentile, axis=0)  


    def get_reconstruction(self):
        return self.image_est

    def get_acceptanceProbability(self):
        breakpoint()
        A = np.reshape(self.acceptanceRatios, self.signal.shape)
        # Initialize the Hermitian symmetric matrix
        hermitian_symmetric_matrix = np.zeros((self.n1, self.n2))

        # Construct the upper triangular part of the matrix
        hermitian_symmetric_matrix[:, :(self.signal.shape[0] // 2 + 1)] = A

        # Fill the lower triangular part by conjugating the upper triangular part
        hermitian_symmetric_matrix[:, self.signal.shape[0] // 2 + 1:] = np.conj(A[self.signal.shape[0] // 2 - 1:0:-1, :])

        # return np.fft.irfft2(np.reshape(self.acceptanceRatios, self.signal.shape), s=(self.n1, self.n2))

    def getResults(self):
        # Get posterior estimates in image space
        self.polarToCartesianChains()
        self.toComplexChains()
        self.saveOriginValuesChains()
        self.toImageSpaceChains()
        self.setImagePosteriorEstimates()
        
    def runBIFS(self): 
        self.toFourierSpace() # Image Space -> Fourier Space (Hermitian property satisfied). Keep half of Fourier Space (including the origin)
        self.setLikelihoodParameters() # Set Likelihood parameters
        self.setPriorParameters() # Set Prior parameters
        self.setInitialValues() # Set initial values for MCMC
        
        mcmc = Adaptive(
                        self.sample,  # Number of posterior samples
                        self.burn,  # burn period
                        self.thin,  # thinning
                        self.batches, self.batch_size,  # number of batches and their size for adapting
                        self.sigma_q,  # initial values of variances for the proposal distribution
                        self.r, self.psi, self.sigma2,  # observed polar values of the first signal
                        self.mu_rho, self.sigma_rho2,  # Prior distribution parameters for (ρ, θ)
                        self.initial_values,  # Initial values for the MCMC (Polar)
                        self.adapting, # adaptive algorithm (true, false)
                        self.seed  # seed
                       )

        mcmc.run()  # Run MCMC
        chains = mcmc.getChains()  # Save chains for each variable
        self.rho_chains, self.theta_chains = chains[0, :, :], chains[1, :, :]
        self.acceptanceRatios = mcmc.getAcceptanceRatios()  # Save acceptance Ratios            
        self.getResults() # Save estimates
            
        
            
