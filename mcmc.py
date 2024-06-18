import numpy as np
import Utils.tools as utils

def log_acceptanceRatio(rho_prop, theta_prop, rho_prev, theta_prev, r, psi, sigma2, mu_rho, sigma_rho2):
    A = (sigma2 + sigma_rho2) / (2 * sigma2 * sigma_rho2)
    B_theta_prop = (r * np.cos(psi - theta_prop) / sigma2) + (mu_rho / sigma_rho2)
    B_theta_prev = (r * np.cos(psi - theta_prev) / sigma2) + (mu_rho / sigma_rho2)
    sq_diff = A * (rho_prev**2 - rho_prop**2)
    diff = B_theta_prop * rho_prop - B_theta_prev * rho_prev
    J_diff = np.log(np.abs(rho_prev)) - np.log(np.abs(rho_prop))
    ratio = sq_diff + diff + J_diff
    return ratio

class MCMC:

    def __init__(self, sample, burn, thin, sigma_q, r, psi, sigma2, mu_rho, sigma_rho2, initial_values, seed):
        #------------------------------------------------------------#
        # sample: Number of posterior samples
        # burn: burn period
        # thin: thinning
        # sigma_q: Variances for the proposal distribution
        # r, psi, sigma2: observed polar values of the first signal
        # mu_rho, sigma_rho2: Prior distribution parameters for (ρ, θ)
        # initial_values: Initial values for the MCMC (Polar)
        # seed: seed
        #------------------------------------------------------------#
        
        self.acceptedSteps = None
        self.acceptedSteps = 0
        self.seed = seed

        # MCMC parameters
        self.sample = sample
        self.burn = burn
        self.thin = thin

        # Initialize standard deviation of proposals steps for (ρ, θ)
        self.sigma_q = sigma_q

        # Observed values
        self.r = r
        self.psi = psi
        self.sigma2 = sigma2

        # Prior parameters for (ρ1, θ1)
        self.mu_rho = mu_rho
        self.sigma_rho2 = sigma_rho2

        # Initial values for chains (ρ1, θ1, Δρ, Δθ)
        self.initial_values = initial_values

        # Number of signals
        self.num_signals = np.size(self.r)

        # Matrix with posterior samples of the polar coordinates
        self.chains = np.zeros((2, self.sample, self.num_signals))

    def get_chains(self):
        return self.chains

    def get_acceptanceRates(self):
        return self.acceptedSteps / self.sample

    def getChainsLastValues(self):
        return [self.chains[0, self.sample - 1, :], self.chains[1, self.sample - 1, :]]

    def propose(self, a, b, sigma):
        a_prop = np.random.normal(a, sigma, self.num_signals)  # Propose real value
        b_prop = np.random.normal(b, sigma, self.num_signals)  # Propose imaginary value
        polar = utils.cartesianToPolar(a_prop, b_prop)  # Transform to polar and return
        return polar

    def run(self):

        np.random.seed(self.seed)  # Set seed

        # Save initial values of polar coordinates in the chain
        rho_prev, theta_prev = self.initial_values
        a_prev, b_prev = utils.polarToCartesian(rho_prev, theta_prev)

        self.acceptedSteps = np.zeros(self.num_signals)  # Accepted steps for (ρ, θ)

        iters = 0  # Counter of total iterations
        size = 0  # number of values inside chain
        
        # If there is no burn save initial states to the chains
        if self.burn == 0:
            self.chains[0, 0, :] = rho_prev
            self.chains[1, 0, :] = theta_prev
            size += 1

        while True:

            # Propose values for (ρ, θ)
            rho_prop, theta_prop = self.propose(a_prev, b_prev, self.sigma_q)

            # Compute the np.logarithm of the acceptance probability for (ρ, θ)
            R = log_acceptanceRatio(rho_prop, theta_prop, rho_prev, theta_prev, self.r, self.psi, self.sigma2, self.mu_rho, self.sigma_rho2)

            log_acceptance = np.where(R >= np.zeros(self.num_signals), 0, R)

            # Accept or reject move
            rho_new = np.where(np.log(np.random.uniform(0, 1, self.num_signals)) < log_acceptance, rho_prop, rho_prev)
            theta_new = np.where(np.log(np.random.uniform(0, 1, self.num_signals)) < log_acceptance, theta_prop, theta_prev)

            # Save updated values in a list
            updatedValues = [rho_new, theta_new]

            if iters >= self.burn and size % self.thin == 0:

                # Update number of accepted steps
                self.acceptedSteps += np.where(rho_new != rho_prev, 1, 0)
                # self.chains[size, :, :] = updatedValues  # Update chains
                self.chains[0, size, :] = rho_new
                self.chains[1, size, :] = theta_new
                size += 1

            # Save initial values as previous for the next iteration of MCMC
            a_prev, b_prev = utils.polarToCartesian(rho_new, theta_new)
            rho_prev, theta_prev = updatedValues

            # Terminate algorithm condition
            if size >= self.sample:
                break

            iters += 1
