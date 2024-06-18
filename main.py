from load_data import LoadData
from BIFS import BIFS
import Utils.figure as F
import matplotlib.pyplot as plt
import numpy as np

image = LoadData(
                 dimensions=2, 
                 slice=66,
                 axis=0,
                 type_noise="gaussian",
                 sd_noise=0.2,
                 isTumor=True,
                 centerTumor=(60, 80, 60),
                 radiusTumor=6,
                 intensity=0.5,
                 seed=1
                )

suboptimal = image.getSuboptimalImage(); true = image.getTrueImage()

bifs = BIFS(suboptimal,  # Input image
            d=1, 
            lamd=2,  # Parameter function for (ρ, θ)
            trim=10, # Value to trim the image and estimate the standard deviation of noise
            sample=1000,  # Number of samples
            burn=0,  # Number of burn samples
            thin=1,  # Thinning
            batches=50,  # Number of batches (for adaptation)
            batch_size=30,  # Iterations for each bach (for adaptation)
            sigma_q=1,  # Variances for the proposal distribution
            adapting=True,  # Adaptive MCMC option
            type_estimation="mean", # Type of method (mean, median, quantile, MAP)
            q_percentile=5, # q-percentile for type_estimation = quantile (manual)
            seed=1
           )

bifs.runBIFS()
reconstruction = bifs.get_reconstruction()

plt.imshow(reconstruction, cmap="gray")
plt.show()








