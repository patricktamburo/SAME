import emcee 
import numpy as np 
import matplotlib.pyplot as plt
plt.ion()
import corner 

cluster_ind = 1
filename = f'/home/ptamburo/tierras/pat_scripts/SAME/output/clusters/{cluster_ind}/cluster{cluster_ind}_mcmc.h5'
reader = emcee.backends.HDFBackend(filename)
samples = reader.get_chain()
flat_samples = reader.get_chain(flat=True)
labels = ['Pos. (pix)', 'FWHM (")', 'Sky (ADU/pix/s)', 'Airmass', 'Norm. Flux', 'Dome Humidity (%)']
cornerfig = corner.corner(flat_samples, labels=labels)

breakpoint()