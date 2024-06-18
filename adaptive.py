import numpy as np
from mcmc import MCMC

class Adaptive:

    def __init__(self,
                 sample,  # Number of posterior samples
                 burn,  # burn period
                 thin,  # thinning
                 batches, batch_size,  # number of batches and their size for adapting
                 s_init,  # initial values of variances for the proposal distribution
                 r, psi, sigma2,  # observed polar values of the first signal
                 mu_rho, sigma_rho2,  # Prior distribution parameters for (ρ, θ)
                 initial_values,  # Initial values for the MCMC (Polar)
                 adaptive,
                 seed  # seed
                 ):

        self.getAcceptanceRates = None
        self.chains = None

        self.sample = sample
        self.burn = burn
        self.thin = thin
        self.batches = batches
        self.batch_size = batch_size
        self.s_init = s_init
        self.adaptive = adaptive

        self.r = r
        self.psi = psi
        self.sigma2 = sigma2

        self.mu_rho = mu_rho
        self.sigma_rho2 = sigma_rho2

        self.initial_values = initial_values
        self.num_signals = np.size(self.r)
        self.seed = seed

    def getChains(self):
        return self.chains

    def getAcceptanceRatios(self):
        return self.getAcceptanceRates

    def run(self):

        np.random.seed(self.seed)
        s_prev = self.s_init
        initial_values = self.initial_values

        if self.adaptive:
            print("Adapting Period...")
            for batch in range(self.batches):
                
                adaptMCMC = MCMC(
                                 self.batch_size,  # Number of posterior samples
                                 0,  # burn period
                                 self.thin,  # thinning
                                 s_prev,  # Variances for the proposal distribution
                                 self.r, self.psi, self.sigma2,  # observed polar values of the first signal
                                 self.mu_rho, self.sigma_rho2,  # Prior distribution parameters for (ρ, θ)
                                 initial_values,  # Initial values for the MCMC (Polar)
                                 self.seed  # seed
                                )
                adaptMCMC.run()
                initial_values = adaptMCMC.getChainsLastValues()
                acceptanceRate = adaptMCMC.get_acceptanceRates()
                delta = (batch + 1) ** (-0.5)

                logs_prev = np.log(s_prev)
                logs_new = logs_prev + np.where(acceptanceRate > np.full(self.num_signals, 0.234), delta, -delta)
                s_new = np.exp(logs_new)
                s_prev = s_new

            # Save optimal standard deviation after the adapting period
            s_optim = s_prev
            print("Sampling Period...")
            sampleMCMC = MCMC(
                              self.sample,  # Number of posterior samples
                              self.burn,  # burn period
                              self.thin,  # thinning
                              s_optim,  # Variances for the proposal distribution
                              self.r, self.psi, self.sigma2,  # observed polar values of the first signal
                              self.mu_rho, self.sigma_rho2,  # Prior distribution parameters for (ρ, θ)
                              initial_values,  # Initial values for the MCMC (Polar)
                              self.seed  # seed
                             )

            # Run the mcmc
            sampleMCMC.run()

        else:
            print("Sampling Period...")
            sampleMCMC = MCMC(
                              self.sample,  # Number of posterior samples
                              self.burn,  # burn period
                              self.thin,  # thinning
                              self.s_init,  # Variances for the proposal distribution
                              self.r, self.psi, self.sigma2,  # observed polar values of the first signal
                              self.mu_rho, self.sigma_rho2,  # Prior distribution parameters for (ρ, θ)
                              initial_values,  # Initial values for the MCMC (Polar)
                              self.seed  # seed
                             )

            # Run the mcmc
            sampleMCMC.run()

        # Get the acceptance ratios
        self.getAcceptanceRates = sampleMCMC.get_acceptanceRates()
        # Save the chains (polar)
        self.chains = sampleMCMC.get_chains()
