import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.image as mpimg

def image(matrix, color):
    if color:
        plt.imshow(matrix, cmap="CMRmap")
        im_ratio = np.shape(matrix)[0] / np.shape(matrix)[1]
        cbar = plt.colorbar(fraction=0.047 * im_ratio)
        cbar.set_label('prob')
    else:
        plt.imshow(matrix, cmap="gray")
        im_ratio = np.shape(matrix)[0] / np.shape(matrix)[1]
        cbar = plt.colorbar(fraction=0.047 * im_ratio)
        cbar.set_label('prob')
        # plt.title("Noisy image", fontsize= 15)
    plt.axis('off')
    plt.show()

def plotEstimates(estimates):
    minim = np.min(estimates)
    maxim = np.max(estimates)

    plt.figure(figsize=(12, 7))
    plt.subplot(1, 3, 1)
    plt.title("Mean", fontsize=15)
    plt.imshow(estimates[0, :, :], cmap='gray', vmin=minim, vmax=maxim)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("Mode", fontsize=15)
    plt.imshow(estimates[1, :, :], cmap='gray', vmin=minim, vmax=maxim)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Median", fontsize=15)
    plt.imshow(estimates[2, :, :], cmap='gray', vmin=minim, vmax=maxim)
    plt.axis('off')
    plt.show()

def plotDeviations(estimates):
    posterior_std = estimates[0,:,:]
    posterior_interquantile_range = estimates[1,:,:]

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title("Posterior SD", fontsize=15)
    a = plt.imshow(posterior_std, cmap="CMRmap")
    plt.colorbar(a, orientation='horizontal')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Posterior IQR", fontsize=15)
    b = plt.imshow(posterior_interquantile_range, cmap="CMRmap")
    plt.colorbar(b, orientation='horizontal')
    plt.axis('off')

    plt.show()

def plotReconstructions(reconstructions, sd_noises, estimates):
    minim = np.min(reconstructions) 
    maxim = np.max(reconstructions)

    y_titles = [0,3,6]
    x_titles = [0,1,2]
    k=0
    l=0
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(reconstructions[i,:,:], cmap="gray", vmin=minim, vmax=maxim)
        if i in y_titles:
            txt = r'$\sigma_{\mathrm{noise}}$ =' + str(sd_noises[k])
            plt.text(-45.5, len(reconstructions[i,:,:])/2, txt, rotation=0, ha='center', va='center', fontsize=15)
            k+=1
        if i in x_titles:
            plt.title(estimates[l], fontsize=15)
            l+=1
        plt.axis('off')
    plt.show()

def plotReconstructions2(reconstructions, estimates):
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        a = plt.imshow(reconstructions[i,:,:], cmap="gray", vmin=np.min(reconstructions), vmax=np.max(reconstructions))
        plt.title(estimates[i], fontsize=15)
        plt.axis('off')
    plt.show()

def plotParamFunc(results, lambdas, ds):
    minim = np.min(results)
    maxim = np.max(results)
    i = 0
    y_titles = [0,3,6]
    x_titles = [0,1,2]
    k=0
    l=0
    plt.figure(figsize=(8,8))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(results[i,:,:], cmap="gray", vmin=minim, vmax=maxim)
        if i in y_titles:
            plt.text(-40.5, len(results[i,:,:])/2, "d = {}".format(ds[k]), rotation=0, ha='center', va='center', fontsize=15)
            k+=1
        if i in x_titles:
            plt.title(r'$\lambda$ = {}'.format(lambdas[l]), fontsize=15)
            l+=1
        plt.axis('off')
    plt.show()

def plotProbabilityMap(probability_map):
    plt.imshow(probability_map, cmap="CMRmap")
    im_ratio = np.shape(probability_map)[0] / np.shape(probability_map)[1]
    cbar = plt.colorbar(fraction=0.047 * im_ratio)
    plt.show()
    cbar.set_label('prob', fontsize=20)


def traceplot(chain, signal, variable):
    plt.plot(chain[:, signal], color="black")
    plt.xlabel("Iterations")
    plt.ylabel(variable)
    title = variable + "traceplot"
    plt.title(title)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    plt.gca()
    plt.show()


def acceptanceProbabilities(acc_probs):
    plt.imshow(acc_probs, cmap="CMRmap")
    im_ratio = np.shape(acc_probs)[0] / np.shape(acc_probs)[1]
    cbar = plt.colorbar(fraction=0.047 * im_ratio)
    cbar.set_label('prob')
    plt.show()

def residuals(true, estimated):
    res = np.absolute(true - estimated)
    mse = np.mean(res)
    rmse = np.sqrt(mse)
    print(rmse)
    plt.imshow(res, cmap="CMRmap")
    im_ratio = np.shape(res)[0] / np.shape(res)[1]
    cbar = plt.colorbar(fraction=0.047 * im_ratio)
    plt.title("Residuals", fontsize=15)
    plt.axis('off')
    plt.show()

def results(suboptimal1, suboptimal2, reconstruction1, reconstruction2, probability_map):
    plt.figure(figsize=(16, 16))

    plt.subplot(2, 3, 1)
    plt.title("Noisy 1", fontsize=20)
    plt.imshow(suboptimal1, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Noisy 2", fontsize=20)
    plt.imshow(suboptimal2, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Probability Map Ψ", fontsize=20)
    plt.imshow(probability_map, cmap='CMRmap')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Reconstructed 1", fontsize=20)
    plt.imshow(reconstruction1, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Reconstructed 2", fontsize=20)
    plt.imshow(reconstruction2, cmap='gray')
    plt.axis('off')

    plt.show()


def mask(anatomical, probability_map):
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Display the original image in grayscale
    ax.imshow(anatomical, cmap='gray')
    # Define your own colormap for the probability map
    cmap = plt.get_cmap('coolwarm')
    # Set the vmin and vmax values based on your probability map range
    vmin, vmax = 0, 1
    # Overlay the probability map with a color map
    cax = ax.imshow(probability_map, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8)
    # Add a color bar to the side of the plot
    cbar = fig.colorbar(cax)
    cbar.set_label('probability of change', fontsize=15)
    plt.axis('off')
    # Show the plot
    plt.show()


def plotMAPSimulation(true, suboptimal, reconstructed):
    plt.figure(figsize=(16, 16))

    plt.subplot(2, 2, 1);
    plt.title("True image", fontsize=20)
    plt.imshow(true, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2);
    plt.title("Suboptimal image\n (truth + noise)", fontsize=20)
    plt.imshow(suboptimal, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3);
    plt.title("Reconstructed image\n (BIFS)", fontsize=20)
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')


def plotMAPReal(observed, reconstructed):
    plt.figure(figsize=(16, 16))

    plt.subplot(1, 2, 1);
    plt.title("Observed image", fontsize=20)
    plt.imshow(observed, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2);
    plt.title("Reconstructed image\n (BIFS)", fontsize=20)
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')


def plotAll(postMap1, postMap2, postMapDiff, postMean1, postMean2, postMeanDiff, accept1, accept2, probMap):
    plt.figure(figsize=(16, 16))

    plt.subplot(3, 3, 1);
    plt.title("Posterior MAP 1", fontsize=20)
    plt.imshow(postMap1, cmap='gray')

    plt.subplot(3, 3, 2);
    plt.title("Posterior MAP 2", fontsize=20)
    plt.imshow(postMap2, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 3);
    plt.title("Posterior MAP\n Difference", fontsize=20)
    plt.imshow(postMapDiff, cmap='CMRmap')
    im_ratio = np.shape(postMapDiff)[0] / np.shape(postMapDiff)[1]
    cbar = plt.colorbar(fraction=0.047 * im_ratio)
    cbar.set_label('prob')
    plt.axis('off')

    plt.subplot(3, 3, 4);
    plt.title("Posterior mean 1", fontsize=20)
    plt.imshow(postMean1, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 5);
    plt.title("Posterior mean 2", fontsize=20)
    plt.imshow(postMean2, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 6);
    plt.title("Posterior mean\n Differences", fontsize=20)
    plt.imshow(postMeanDiff, cmap='CMRmap')
    im_ratio = np.shape(postMeanDiff)[0] / np.shape(postMeanDiff)[1]
    cbar = plt.colorbar(fraction=0.047 * im_ratio)
    cbar.set_label('prob')
    plt.axis('off')

    plt.subplot(3, 3, 7);
    plt.title("Acceptance Ratio 1\n Fourier Space", fontsize=20)
    plt.imshow(accept1, cmap='CMRmap')
    im_ratio = np.shape(accept1)[0] / np.shape(accept1)[1]
    cbar = plt.colorbar(fraction=0.047 * im_ratio)
    cbar.set_label('prob')
    plt.axis('off')

    plt.subplot(3, 3, 8);
    plt.title("Acceptance Ratio Differences\n Fourier Space", fontsize=20)
    plt.imshow(accept2, cmap='CMRmap')
    im_ratio = np.shape(accept2)[0] / np.shape(accept2)[1]
    cbar = plt.colorbar(fraction=0.047 * im_ratio)
    cbar.set_label('prob')
    plt.axis('off')

    plt.subplot(3, 3, 9);
    plt.title("Probability map", fontsize=20)
    plt.imshow(probMap, cmap='CMRmap')
    im_ratio = np.shape(probMap)[0] / np.shape(probMap)[1]
    cbar = plt.colorbar(fraction=0.047 * im_ratio)
    cbar.set_label('prob')
    plt.axis('off')
