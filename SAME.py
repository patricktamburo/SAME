import lightkurve as lk
import emcee
import corner
import warnings
import eleanor as el 
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.sdss import SDSS
from astropy.visualization import simple_norm, ZScaleInterval, ImageNormalize
from astropy.modeling.functional_models import Gaussian2D
import numpy as np 
import matplotlib.pyplot as plt
plt.ion()
from astropy.io import fits
from astropy.coordinates import SkyCoord,Distance
import astropy.units as u 
from astropy.time import Time
from astropy.wcs import WCS
import matplotlib.colors as mcolors
import astroplan
from glob import glob 
from ap_phot import get_flattened_files, load_bad_pixel_mask, jd_utc_to_bjd_tdb, plot_image, set_tierras_permissions, tierras_binner
import os 
import sep 
import time
from scipy.stats import sigmaclip, kstest
from photutils.aperture import CircularAperture, EllipticalAperture, CircularAnnulus, aperture_photometry
from photutils.background import Background2D, MedianBackground
from pathlib import Path 
import pandas as pd 
import copy 
from astropy.modeling import models, fitting
from photutils.centroids import centroid_sources, centroid_1dg, centroid_2dg, centroid_com, centroid_quadratic
from astropy.stats import sigma_clipped_stats, SigmaClip
from scipy.interpolate import interp1d 
from astropy.table import Table 
import logging 

def circular_footprint(radius, dtype=int):
    """
	TAKEN DIRECTLY FROM PHOTUTILS!!!
    Create a circular footprint.

    A pixel is considered to be entirely in or out of the footprint
    depending on whether its center is in or out of the footprint.  The
    size of the output array is the minimal bounding box for the
    footprint.

    Parameters
    ----------
    radius : int
        The radius of the circular footprint.

    dtype : data-type, optional
        The data type of the output `~numpy.ndarray`.

    Returns
    -------
    footprint : `~numpy.ndarray`
        A footprint where array elements are 1 within the footprint and
        0 otherwise.

    Examples
    --------
    >>> from photutils.utils import circular_footprint
    >>> circular_footprint(2)
    array([[0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0]])
    """
    if ~np.isfinite(radius) or radius <= 0 or int(radius) != radius:
        raise ValueError('radius must be a positive, finite integer greater '
                         'than 0')

    x = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, x)
    return np.array((xx**2 + yy**2) <= radius**2, dtype=dtype)

def median_standard_error(data, n_bootstrap=5000):
	# generate a len(data) x n_bootstrap array of random samples from the input data
	bootstrap_samples = np.random.choice(data, size=(len(data), n_bootstrap), replace=True)

	# get the median of each sample 
	bootstrap_medians = np.nanmedian(bootstrap_samples, axis=0)

	# return the std of bootstrap_medians as an estimate of the standard error on the median
	return np.nanstd(bootstrap_medians)

def generate_square_cutout(image, position, size):
	height, width = image.shape
	
	# Calculate the bounds of the cutout
	half_size = size / 2
	top = max(0, int(position[0] - half_size))
	bottom = min(width, int(position[0] + half_size))
	left = max(0, int(position[1] - half_size))
	right = min(height, int(position[1] + half_size))
	
	# Calculate the actual position within the cutout
	adjusted_position = (int(half_size - (position[0] - top)), int(half_size - (position[1] - left)))
	
	# Generate the square cutout
	cutout = np.zeros((size, size), dtype=image.dtype)
	cutout[adjusted_position[1]:(adjusted_position[1] + right - left), adjusted_position[0]:(adjusted_position[0] + bottom - top),] = image[left:right, top:bottom]

	cutout[cutout == 0] = np.nan
	
	# Calculate the position of the input 'position' in the cutout
	# position_in_cutout = (half_size - adjusted_position[0], half_size - adjusted_position[1])
	position_in_cutout = (adjusted_position[0] + position[0] - top, adjusted_position[1] + position[1] - left)

	

	return cutout, position_in_cutout

def create_linked_tuples(tuple_list):
	# Sort the list of tuples based on the second element
	sorted_tuples = sorted(tuple_list, key=lambda x: x[1])
	
	# Initialize an empty list to store the resulting linked tuples
	linked_tuples = []
	
	# Iterate through the sorted list to build the chains of tuples
	i = 0
	while i < len(sorted_tuples) - 1:
		# Initialize a list to store the linked elements
		chain = [sorted_tuples[i][0]]
		
		# Find all linked elements with the same second element
		j = i + 1
		while j < len(sorted_tuples) and sorted_tuples[j][1] == sorted_tuples[i][1]:
			chain.append(sorted_tuples[j][0])
			j += 1
		
		# If there are more than one linked elements, create a new tuple and append it to the result
		if len(chain) > 1:
			chain.append(sorted_tuples[i][1])
			linked_tuples.append(tuple(chain))
		
		# Move to the next group of linked elements
		i = j
	
	return linked_tuples

def is_subset(tuple1, tuple2):
    """Check if tuple1 is a subset of tuple2."""
    return all(elem in tuple2 for elem in tuple1)

def gaia_query(field, bp_rp_threshold=0.2, G_threshold=0.5, rp_threshold=0.5, distance_threshold=4223, edge_threshold=30, same_chip=True, max_rp=16.5, overwrite=False, contamination_limit = 0.001):
	''' For a given Tierras field, identify stars that are similar types (Teff/logg), on the same chip, and are not variable in TESS'''

	# check for existing sources.csv output, if it exists then restore/return	
	if (os.path.exists('/home/ptamburo/tierras/pat_scripts/SAME/output/sources/sources.csv')) and (not overwrite):
		print('Restoring sources.csv!')
		stars = Table.from_pandas(pd.read_csv('/home/ptamburo/tierras/pat_scripts/SAME/output/sources/sources.csv'))
		return stars

	PLATE_SCALE = 0.432 
	stacked_image_path = f'/data/tierras/targets/{field}/{field}_stacked_image.fits'
	hdu = fits.open(stacked_image_path)
	data = hdu[0].data
	header = hdu[0].header
	wcs = WCS(header)
	tierras_epoch = Time(header['TELDATE'],format='decimalyear')
	ra = header['RA']
	dec = header['DEC']
	# fig, ax = plot_image(data)
	fig, ax = plt.subplots(1,1,figsize=(13,8))
	ax.imshow(data, origin='lower', cmap='Greys_r', norm=simple_norm(data, min_percent=1,max_percent=99))
	plt.tight_layout()

	#coord = SkyCoord(ra,dec,unit=(u.hourangle,u.degree),frame='icrs')
	coord = SkyCoord(wcs.pixel_to_world(int(data.shape[1]/2),int(data.shape[0]/2)))
	#width = u.Quantity(PLATE_SCALE*data.shape[0],u.arcsec)
	width = 1750*u.arcsec
	height = u.Quantity(PLATE_SCALE*data.shape[1],u.arcsec)

	job = Gaia.launch_job_async("""SELECT
								source_id, ra, dec, ref_epoch, pmra, pmra_error, pmdec, pmdec_error, parallax, parallax_error, ruwe, phot_bp_mean_mag, phot_g_mean_mag, phot_rp_mean_mag, bp_rp, phot_variable_flag, non_single_star,
									teff_gspphot, logg_gspphot, mh_gspphot, ag_gspphot,	ebpminrp_gspphot, 
									r_med_geo, r_lo_geo, r_hi_geo,
									r_med_photogeo, r_lo_photogeo, r_hi_photogeo,
									phot_bp_mean_mag-phot_rp_mean_mag AS bp_rp,
									phot_g_mean_mag - 5 * LOG10(r_med_geo) + 5 AS qg_geo,
									phot_g_mean_mag - 5 * LOG10(r_med_photogeo) + 5 AS gq_photogeo
									FROM (
										SELECT * FROM gaiadr3.gaia_source as gaia

										WHERE gaia.ra BETWEEN {} AND {} AND
											  gaia.dec BETWEEN {} AND {} 
										OFFSET 0
									) AS edr3
									JOIN external.gaiaedr3_distance using(source_id)
									ORDER BY phot_rp_mean_mag ASC
								""".format(coord.ra.value-width.to(u.deg).value/2, coord.ra.value+width.to(u.deg).value/2, coord.dec.value-height.to(u.deg).value/2, coord.dec.value+height.to(u.deg).value/2)
								)
	res = job.get_results()
	
	#Cut to entries without masked pmra values; otherwise the crossmatch will break
	problem_inds = np.where(res['pmra'].mask)[0]

	#OPTION 1: Set the pmra, pmdec, and parallax of those indices to 0
	res['pmra'][problem_inds] = 0
	res['pmdec'][problem_inds] = 0
	res['parallax'][problem_inds] = 0

	# cross-match with 2MASS
	gaia_coords = SkyCoord(ra=res['ra'], dec=res['dec'], pm_ra_cosdec=res['pmra'], pm_dec=res['pmdec'], obstime=Time('2016',format='decimalyear'))
	v = Vizier(catalog="II/246",columns=['*','Date'], row_limit=-1)
	twomass_res = v.query_region(coord, width=width, height=height)[0]
	twomass_coords = SkyCoord(twomass_res['RAJ2000'],twomass_res['DEJ2000'])
	twomass_epoch = Time('2000-01-01')
	gaia_coords_tm_epoch = gaia_coords.apply_space_motion(twomass_epoch)
	gaia_coords_tierras_epoch = gaia_coords.apply_space_motion(tierras_epoch)

	idx_gaia, sep2d_gaia, _ = gaia_coords_tm_epoch.match_to_catalog_sky(twomass_coords)
	#Now set problem indices back to NaNs
	res['pmra'][problem_inds] = np.nan
	res['pmdec'][problem_inds] = np.nan
	res['parallax'][problem_inds] = np.nan
	
	# calculate tierras pixel positions
	tierras_pixel_coords = wcs.world_to_pixel(gaia_coords_tierras_epoch)

	res.add_column(twomass_res['_2MASS'][idx_gaia],name='2MASS',index=1)
	res.add_column(tierras_pixel_coords[0],name='X pix', index=2)
	res.add_column(tierras_pixel_coords[1],name='Y pix', index=3)
	res.add_column(gaia_coords_tierras_epoch.ra, name='ra_tierras', index=4)
	res.add_column(gaia_coords_tierras_epoch.dec, name='dec_tierras', index=5)
	res['Jmag'] = twomass_res['Jmag'][idx_gaia]
	res['e_Jmag'] = twomass_res['e_Jmag'][idx_gaia]
	res['Hmag'] = twomass_res['Hmag'][idx_gaia]
	res['e_Hmag'] = twomass_res['e_Hmag'][idx_gaia]
	res['Kmag'] = twomass_res['Kmag'][idx_gaia]
	res['e_Kmag'] = twomass_res['e_Kmag'][idx_gaia]

	# determine which chip the sources fall on 
	# 0 = bottom, 1 = top 
	chip_inds = np.zeros(len(res),dtype='int')
	chip_inds[np.where(res['Y pix'] >= 1023)] = 1
	res.add_column(chip_inds, name='Chip')

	# save a copy of all the sources before we start cutting them 

	#Cut to sources that actually fall in the image
	use_inds = np.where((tierras_pixel_coords[0]>0)&(tierras_pixel_coords[0]<data.shape[1]-1)&(tierras_pixel_coords[1]>0)&(tierras_pixel_coords[1]<data.shape[0]-1))[0]
	res = res[use_inds]
	res_full = copy.deepcopy(res)	
	print(f'Found {len(res)} sources in Gaia DR3!')

	# remove sources near the bad columns 
	bad_inds = np.where((res['Y pix'] < 1035) & (res['X pix'] > 1434) & (res['X pix'] < 1469))[0]
	res.remove_rows(bad_inds)

	bad_inds = np.where((res['Y pix'] > 997) & (res['X pix'] > 1776) & (res['X pix'] < 1810))[0]
	res.remove_rows(bad_inds)

	#Cut to sources that are away from the edges
	use_inds = np.where((res['Y pix'] > edge_threshold) & (res['Y pix']<data.shape[0]-edge_threshold-1) & (res['X pix'] > edge_threshold) & (res['X pix'] < data.shape[1]-edge_threshold-1))[0]
	res = res[use_inds]
	
	# remove sources flagged as variable
	use_inds =np.where(res['phot_variable_flag'] != 'VARIABLE')[0]
	res = res[use_inds]

	# remove sources flagged as non-single stars 
	use_inds = np.where(res['non_single_star'] == 0)[0]
	res = res[use_inds] 

	# remove sources with high RUWE values 
	use_inds = np.where(res['ruwe'] < 1.4)[0]
	res = res[use_inds]

	# remove faint sources 
	use_inds = np.where(res['phot_rp_mean_mag'] < 18)[0]
	res = res[use_inds]

	print(f'{len(res)} sources remain after initial cuts')
	
	inds_to_remove = []
	xx, yy = np.meshgrid(np.arange(-100, 100), np.arange(-100, 100)) # grid of pixels over which to simulate images
	seeing_fwhm = 3/.432 #pix, assume 3" seeing
	seeing_sigma = seeing_fwhm / (2*np.sqrt(2*np.log(2)))


	for i in range(len(res)):
		print(f'Estimating contamination for source {i+1} of {len(res)}')
		distances = ((res['X pix'][i]-res_full['X pix'])**2 + (res['Y pix'][i]-res_full['Y pix'])**2)**0.5
		nearby_inds = np.where(distances <= 100)[0]
		nearby_inds = nearby_inds[np.where(distances[nearby_inds]!=0)[0]]
		if len(nearby_inds) > 0:
			source_rp = res['phot_rp_mean_mag'][i]
			source_x = res['X pix'][i]
			source_y = res['Y pix'][i]
			nearby_rp = res_full['phot_rp_mean_mag'][nearby_inds]
			nearby_x = res_full['X pix'][nearby_inds] - source_x
			nearby_y = res_full['Y pix'][nearby_inds] - source_y 

			# model the source in question as a 2D gaussian
			source_model = Gaussian2D(x_mean=0, y_mean=0, amplitude=1/(2*np.pi*seeing_sigma**2), x_stddev=seeing_sigma, y_stddev=seeing_sigma)
			sim_img = source_model(xx, yy)

			# add in gaussian models for the nearby sources
			for jj in range(len(nearby_rp)):
				flux = 10**(-(nearby_rp[jj]-source_rp)/2.5)
				contaminant_model = Gaussian2D(x_mean=nearby_x[jj], y_mean=nearby_y[jj], amplitude=flux/(2*np.pi*seeing_sigma**2), x_stddev=seeing_sigma, y_stddev=seeing_sigma)
				contaminant = contaminant_model(xx,yy)
				sim_img += contaminant
			
			# plt.figure()
			# plt.imshow(sim_img, origin='lower', norm=simple_norm(sim_img, min_percent=1, max_percent=80))
			
			# estimate contamination by doing aperture photometry on the simulated image
			# if the measured flux exceeds 1 by a chosen threshold, record the source's index so it can be removed
			ap = CircularAperture((sim_img.shape[1]/2, sim_img.shape[0]/2), r=2*seeing_fwhm)
			phot_table = aperture_photometry(sim_img, ap)
			contamination = phot_table['aperture_sum'][0] - 1
			if contamination > contamination_limit:
				inds_to_remove.append([i])
	res.remove_rows(inds_to_remove)

	# ax.plot(res_full['X pix'], res_full['Y pix'], 'bx')
	# ax.plot(res['X pix'], res['Y pix'], 'rx')
	
	print(f'{len(res)} sources remain after contamination cuts')	
	
	# remove non-main-sequence sources
	upper_ms_bp_rp = np.arange(min(res_full['bp_rp']),max(res_full['bp_rp']),0.01) # create an estimate of the upper main sequence to exclude giants
	upper_ms = 2.1 + 2.9*upper_ms_bp_rp # determined by-eye 
	bad_inds = []
	for i in range(len(res)):
		bp_rp = res['bp_rp'][i]
		model_bp_rp_ind = np.argmin(abs(upper_ms_bp_rp-bp_rp))
		if res['gq_photogeo'][i] < upper_ms[model_bp_rp_ind]:
			bad_inds.append(i)
	res.remove_rows(bad_inds)
	print(f'{len(res)} sources remain after main sequence cuts')

	ax.plot(res['X pix'], res['Y pix'], marker='x', ls='', color='tab:red')

	fig1, ax1 = plt.subplots(1,2,figsize=(10,5)) 
	ax1[0].scatter(res_full['bp_rp'],res_full['gq_photogeo'],alpha=0.4, color='#b0b0b0', marker='.')
	ax1[0].scatter(res['bp_rp'],res['gq_photogeo'],alpha=0.4, color='tab:blue', marker='.')

	ax1[0].invert_yaxis()
	ax1[0].set_xlabel('B$_p$-R$_p$',fontsize=14)
	ax1[0].set_ylabel('M$_{G}$',fontsize=14)
	ax1[0].tick_params(labelsize=12)

	ax1[1].scatter(res_full['bp_rp'],res_full['phot_rp_mean_mag'],alpha=0.4, color='#b0b0b0', marker='.')
	ax1[1].scatter(res['bp_rp'],res['phot_rp_mean_mag'],alpha=0.4, color='tab:blue', marker='.')
	ax1[1].invert_yaxis()
	ax1[1].set_xlabel('B$_p$-R$_p$',fontsize=14)
	ax1[1].set_ylabel('m$_{Rp}$',fontsize=14)
	ax1[1].tick_params(labelsize=12)
	plt.tight_layout()

	ax1[0].plot(upper_ms_bp_rp, upper_ms, lw=1., ls='--', color='k')


	df = res.to_pandas()
	df.to_csv('/home/ptamburo/tierras/pat_scripts/SAME/output/sources/sources.csv', index=0)

	# create cmd 
	
	return res

def tess_variability_check(star):
	''' check a target's TESS photometry to evaluate whether it should be retained as a SAME star or not'''

	print(f"Doing Gaia DR3 {star['source_id']}")

	# set up a SkyCoord object using the position/proper motion information from Gaia
	ra_gaia = star['ra']
	dec_gaia = star['dec']
	epoch_gaia = star['ref_epoch']
	pmra_gaia = star['pmra']
	pmdec_gaia = star['pmdec']
	sc_gaia = SkyCoord(ra_gaia*u.deg, dec_gaia*u.deg, pm_ra_cosdec=pmra_gaia*u.mas/u.yr, pm_dec=pmdec_gaia*u.mas/u.yr, obstime=Time(epoch_gaia, format='decimalyear'))

	# el_star = el.multi_sectors(coords=sc_gaia, sectors='all')
	el_star = el.Source(coords=sc_gaia)
	el_data = el.TargetData(el_star, height=15, width=15, bkg_size=31, do_psf=False, do_pca=True)
	qual_flags = el_data.quality == 0

	plt.figure()
	plt.plot(el_data.time[qual_flags], el_data.all_corr_flux[0][qual_flags])
	breakpoint()
	plt.close()
	if np.std(el_data.all_corr_flux[0][qual_flags]) > 3*np.mean(el_data.flux_err):

		breakpoint()
	

def star_selection(target, res, bp_rp_threshold=0.2, G_threshold=0.5, rp_threshold=0.5, distance_threshold=4223, edge_threshold=30, same_chip=True, max_rp=16.5):
	''' For a given Tierras field, identify stars that are similar types (Teff/logg), on the same chip, and are not variable in TESS'''
	
	PLATE_SCALE = 0.432 
	stacked_image_path = f'/data/tierras/targets/{target}/{target}_stacked_image.fits'
	hdu = fits.open(stacked_image_path)
	data = hdu[0].data
	header = hdu[0].header
	wcs = WCS(header)
	tierras_epoch = Time(header['TELDATE'],format='decimalyear')
	ra = header['RA']
	dec = header['DEC']
	# fig, ax = plot_image(data)
	fig, ax = plt.subplots(1,1,figsize=(13,8))
	ax.imshow(data, origin='lower', cmap='Greys_r', norm=simple_norm(data, min_percent=1,max_percent=99))
	plt.tight_layout()

	#coord = SkyCoord(ra,dec,unit=(u.hourangle,u.degree),frame='icrs')
	coord = SkyCoord(wcs.pixel_to_world(int(data.shape[1]/2),int(data.shape[0]/2)))
	#width = u.Quantity(PLATE_SCALE*data.shape[0],u.arcsec)
	width = 1500*u.arcsec
	height = u.Quantity(PLATE_SCALE*data.shape[1],u.arcsec)

	# job = Gaia.launch_job_async("""SELECT
	# 							source_id, ra, dec, ref_epoch, pmra, pmra_error, pmdec, pmdec_error, parallax, parallax_error, phot_bp_mean_mag, phot_g_mean_mag, phot_rp_mean_mag, bp_rp, phot_variable_flag, non_single_star,
	# 								teff_gspphot, logg_gspphot, mh_gspphot, ag_gspphot,	ebpminrp_gspphot, 
	# 								r_med_geo, r_lo_geo, r_hi_geo,
	# 								r_med_photogeo, r_lo_photogeo, r_hi_photogeo,
	# 								phot_bp_mean_mag-phot_rp_mean_mag AS bp_rp,
	# 								phot_g_mean_mag - 5 * LOG10(r_med_geo) + 5 AS qg_geo,
	# 								phot_g_mean_mag - 5 * LOG10(r_med_photogeo) + 5 AS gq_photogeo
	# 								FROM (
	# 									SELECT * FROM gaiadr3.gaia_source as gaia

	# 									WHERE gaia.ra BETWEEN {} AND {} AND
	# 										  gaia.dec BETWEEN {} AND {} AND
	# 										  gaia.phot_rp_mean_mag <= 20 AND
	# 						 				  gaia.phot_variable_flag <> 'VARIABLE' AND
	# 						 				  gaia.non_single_star = 0

	# 									OFFSET 0
	# 								) AS edr3
	# 								JOIN external.gaiaedr3_distance using(source_id)
	# 								WHERE ruwe<1.4 
	# 								ORDER BY phot_rp_mean_mag ASC
	# 							""".format(coord.ra.value-width.to(u.deg).value/2, coord.ra.value+width.to(u.deg).value/2, coord.dec.value-height.to(u.deg).value/2, coord.dec.value+height.to(u.deg).value/2)
	# 							)
	# res = job.get_results()

	# #Cut to entries without masked pmra values; otherwise the crossmatch will break
	# problem_inds = np.where(res['pmra'].mask)[0]

	# #OPTION 1: Set the pmra, pmdec, and parallax of those indices to 0
	# res['pmra'][problem_inds] = 0
	# res['pmdec'][problem_inds] = 0
	# res['parallax'][problem_inds] = 0

	# # #OPTION 2: Drop them from the table
	# # good_inds = np.where(~res['pmra'].mask)[0]
	# # res = res[good_inds]

	# gaia_coords = SkyCoord(ra=res['ra'], dec=res['dec'], pm_ra_cosdec=res['pmra'], pm_dec=res['pmdec'], obstime=Time('2016',format='decimalyear'))
	# v = Vizier(catalog="II/246",columns=['*','Date'], row_limit=-1)
	# twomass_res = v.query_region(coord, width=width, height=height)[0]
	# twomass_coords = SkyCoord(twomass_res['RAJ2000'],twomass_res['DEJ2000'])
	# twomass_epoch = Time('2000-01-01')
	# gaia_coords_tm_epoch = gaia_coords.apply_space_motion(twomass_epoch)
	# gaia_coords_tierras_epoch = gaia_coords.apply_space_motion(tierras_epoch)

	# idx_gaia, sep2d_gaia, _ = gaia_coords_tm_epoch.match_to_catalog_sky(twomass_coords)
	# #Now set problem indices back to NaNs
	# res['pmra'][problem_inds] = np.nan
	# res['pmdec'][problem_inds] = np.nan
	# res['parallax'][problem_inds] = np.nan
	
	# tierras_pixel_coords = wcs.world_to_pixel(gaia_coords_tierras_epoch)

	# res.add_column(twomass_res['_2MASS'][idx_gaia],name='2MASS',index=1)
	# res.add_column(tierras_pixel_coords[0],name='X pix', index=2)
	# res.add_column(tierras_pixel_coords[1],name='Y pix', index=3)
	# res.add_column(gaia_coords_tierras_epoch.ra, name='ra_tierras', index=4)
	# res.add_column(gaia_coords_tierras_epoch.dec, name='dec_tierras', index=5)
	# res['Jmag'] = twomass_res['Jmag'][idx_gaia]
	# res['e_Jmag'] = twomass_res['e_Jmag'][idx_gaia]
	# res['Hmag'] = twomass_res['Hmag'][idx_gaia]
	# res['e_Hmag'] = twomass_res['e_Hmag'][idx_gaia]
	# res['Kmag'] = twomass_res['Kmag'][idx_gaia]
	# res['e_Kmag'] = twomass_res['e_Kmag'][idx_gaia]

	# # determine which chip the sources fall on 
	# # 0 = bottom, 1 = top 
	# chip_inds = np.zeros(len(res),dtype='int')
	# chip_inds[np.where(res['Y pix'] >= 1023)] = 1
	# res.add_column(chip_inds, name='Chip')
	
	# #Cut to sources that actually fall in the image
	# use_inds = np.where((tierras_pixel_coords[0]>0)&(tierras_pixel_coords[0]<data.shape[1]-1)&(tierras_pixel_coords[1]>0)&(tierras_pixel_coords[1]<data.shape[0]-1))[0]
	# res = res[use_inds]

	# # remove sources near the bad columns 
	# bad_inds = np.where((res['Y pix'] < 1035) & (res['X pix'] > 1434) & (res['X pix'] < 1469))[0]
	# res.remove_rows(bad_inds)

	# bad_inds = np.where((res['Y pix'] > 997) & (res['X pix'] > 1776) & (res['X pix'] < 1810))[0]
	# res.remove_rows(bad_inds)

	# #Cut to sources that are away from the edges
	# use_inds = np.where((res['Y pix'] > edge_threshold) & (res['Y pix']<data.shape[0]-edge_threshold-1) & (res['X pix'] > edge_threshold) & (res['X pix'] < data.shape[1]-edge_threshold-1))[0]
	# res = res[use_inds]

	# all_sources = copy.deepcopy(res)
		
	# # ax.plot(res['X pix'], res['Y pix'], marker='x', ls='', color='tab:red')

	# # remove sources that are within x pixels of any other source
	# dx = res['X pix'][:,np.newaxis]-res['X pix'][np.newaxis,:]
	# dy = res['Y pix'][:,np.newaxis]-res['Y pix'][np.newaxis,:]
	# distance_matrix = (dx**2+dy**2)**0.5
	# mask = np.tril(np.ones(distance_matrix.shape, dtype=bool), k=-1)
	# np.fill_diagonal(distance_matrix, np.nan)
	# distance_matrix[mask] = np.nan
	# bad_inds = np.unique(np.where(distance_matrix < 25))
	# res.remove_rows(bad_inds)

	
	# # ax.plot(res['X pix'], res['Y pix'], marker='x', ls='', color='tab:blue')

	# # create cmd 
	upper_ms_bp_rp = np.arange(min(res['bp_rp']),max(res['bp_rp']),0.01) # create an estimate of the upper main sequence to exclude giants
	upper_ms = 2.1 + 2.9*upper_ms_bp_rp # determined by-eye 
	fig1, ax1 = plt.subplots(1,2,figsize=(10,7)) 
	ax1[0].scatter(res['bp_rp'],res['gq_photogeo'],alpha=0.4, color='k', marker='.')
	ax1[0].plot(upper_ms_bp_rp, upper_ms, lw=1.5)
	ax1[0].invert_yaxis()
	ax1[0].set_xlabel('B$_p$-R$_p$',fontsize=14)
	ax1[0].set_ylabel('M$_{G}$',fontsize=14)
	ax1[0].tick_params(labelsize=12)

	ax1[1].scatter(res['bp_rp'],res['phot_rp_mean_mag'],alpha=0.4, color='k', marker='.')
	ax1[1].invert_yaxis()
	ax1[1].set_xlabel('B$_p$-R$_p$',fontsize=14)
	ax1[1].set_ylabel('m$_{Rp}$',fontsize=14)
	ax1[1].tick_params(labelsize=12)

	# remove faintest sources
	use_inds = np.where(res['phot_rp_mean_mag'] < 18)[0]
	res = res[use_inds]

	# determine pairs of stars that meet threshold values in bp_rp, G, and rp 
	bp_rp_matrix = np.triu(np.abs(res['bp_rp'][:,np.newaxis]-res['bp_rp']))
	G_matrix = np.triu(np.abs(res['gq_photogeo'][:,np.newaxis]-res['gq_photogeo']))
	rp_matrix = np.triu(np.abs(res['phot_rp_mean_mag'][:,np.newaxis]-res['phot_rp_mean_mag']))
	distance_matrix = np.triu(((res['X pix'][:,np.newaxis]-res['X pix'])**2+(res['Y pix'][:,np.newaxis]-res['Y pix'])**2)**0.5)

	triu_inds = np.triu_indices_from(bp_rp_matrix,k=1)

	bp_rp_candidates_ = bp_rp_matrix[triu_inds] < bp_rp_threshold
	bp_rp_candidates = list(zip(triu_inds[0][bp_rp_candidates_], triu_inds[1][bp_rp_candidates_]))

	G_candidates_ = G_matrix[triu_inds] < G_threshold
	G_candidates= list(zip(triu_inds[0][G_candidates_], triu_inds[1][G_candidates_]))

	rp_candidates_ = rp_matrix[triu_inds] < rp_threshold
	rp_candidates = list(zip(triu_inds[0][rp_candidates_], triu_inds[1][rp_candidates_]))

	distance_candidates = distance_matrix[triu_inds] < distance_threshold
	distance_candidates = list(zip(triu_inds[0][distance_candidates], triu_inds[1][distance_candidates]))

	set1 = set(bp_rp_candidates)
	set2 = set(G_candidates)
	set3 = set(rp_candidates)
	set4 = set(distance_candidates)

	common_tuples = list(set1.intersection(set2).intersection(set3).intersection(set4))
	
	# discard any pairs that lie on different chips, those that are too faint, and likely giants
	tuples_to_remove = []
	for i in range(len(common_tuples)):
		if same_chip:
			if res['Chip'][common_tuples[i][0]] != res['Chip'][common_tuples[i][1]]:
				tuples_to_remove.append(i)
		if (res['phot_rp_mean_mag'][common_tuples[i][0]] > max_rp) or (res['phot_rp_mean_mag'][common_tuples[i][1]] > max_rp):
			tuples_to_remove.append(i)
		if (res['logg_gspphot'][common_tuples[i][0]] < 3.5) or (res['logg_gspphot'][common_tuples[i][1]] < 3.5):
			tuples_to_remove.append(i)
		# cut any pairs with soures that lie above our model main sequence
		model_ms_bp_rp_1 = upper_ms[np.argmin(abs(upper_ms_bp_rp-res['bp_rp'][common_tuples[i][0]]))]
		model_ms_bp_rp_2 = upper_ms[np.argmin(abs(upper_ms_bp_rp-res['bp_rp'][common_tuples[i][1]]))]
		if (res['gq_photogeo'][common_tuples[i][0]] < model_ms_bp_rp_1) or (res['gq_photogeo'][common_tuples[i][1]] < model_ms_bp_rp_2):
			tuples_to_remove.append(i)
	

	common_tuples = [common_tuples[i] for i in range(len(common_tuples)) if i not in tuples_to_remove]
	common_tuples = create_linked_tuples(common_tuples)

	# remove any clusters that are contained within another cluster 
	result = []
	for tup1 in common_tuples:
		if not any(is_subset(tup1, tup2) and len(tup1) < len(tup2) for tup2 in common_tuples if tup1 != tup2):
			result.append(tup1)
	common_tuples = result

	print(f'Found {len(common_tuples)} SAME clusters.')

	# cut to the cluster with the most members 
	common_tuples = [common_tuples[np.argmax([len(i) for i in common_tuples])]]
	

	colors = list(mcolors.TABLEAU_COLORS.values())
	markers = ['x', '+', 'v', '^', 'o', 'D', 'H', '<', '>']
	k = 0 
	for i in range(len(common_tuples)):
		x_offset = np.random.uniform(5,10)
		y_offset = np.random.uniform(5,10)
		print(f'Cluster {i}:')
		for j in range(len(common_tuples[i])):

			print(f"    Source ID: {res['source_id'][common_tuples[i][j]]}, Teff: {res['teff_gspphot'][common_tuples[i][j]]:.0f}, logg: {res['logg_gspphot'][common_tuples[i][j]]:.1f}, Bp-Rp: {res['bp_rp'][common_tuples[i][j]]:.2f} , M_G: {res['gq_photogeo'][common_tuples[i][j]]:.2f}, Rp: {res['phot_rp_mean_mag'][common_tuples[i][j]]:.2f}")

			ax1[0].plot(res['bp_rp'][common_tuples[i][j]],res['gq_photogeo'][common_tuples[i][j]], marker=markers[k], color=colors[i%len(colors)], mew=1, ms=10)
		
			ax1[1].plot(res['bp_rp'][common_tuples[i][j]],res['phot_rp_mean_mag'][common_tuples[i][j]], marker=markers[k], color=colors[i%len(colors)], mew=1, ms=10)

			ax.plot(res['X pix'][common_tuples[i][j]], res['Y pix'][common_tuples[i][j]], marker=markers[k], color=colors[i%len(colors)], mew=1, ms=10)
			ax.text(res['X pix'][common_tuples[i][j]]+x_offset, res['Y pix'][common_tuples[i][j]]+y_offset, f'{i}', color=colors[i%len(colors)],fontsize=14)
			# breakpoint()
		
		print('')
		
		if i != 0  and (i+1)%len(colors) == 0:
			k += 1
		

			

	# get list of unique stars that were identified so that we can do photometry on them 
	unique_same_stars = []
	for i in range(len(common_tuples)):
		for j in range(len(common_tuples[i])):
			if common_tuples[i][j] not in unique_same_stars:
				unique_same_stars.append(common_tuples[i][j])
	
	# figure out the SAME clusters that each unique star belongs to
	same_cluster_ids = []
	for i in range(len(unique_same_stars)):
		ids = []
		for j in range(len(common_tuples)):
			if unique_same_stars[i] in common_tuples[j]:
				ids.extend([j])
		same_cluster_ids.append(ids)

	same_res = res[unique_same_stars]
	# same_res.add_column(same_cluster_ids, name='Cluster IDs')

	# # re-index the SAME stars to count from 0
	# for i in range(len(unique_same_stars)):
	# 	for j in range(len(common_tuples)):
	# 		tup = common_tuples[j]
	# 		arr = np.array(list(tup))
	# 		arr[np.where(arr == unique_same_stars[i])[0]] = i 
	# 		common_tuples[j] = tuple(list(arr))

	# # check tess data for the reference stars to toss any variables
	# for i in range(len(same_res)):
	# 	tess_variability_check(same_res[i])
	return same_res, common_tuples, res

def night_selection(field):
	''' For a given Tierras field, identify nights that are similar in focus/image quality, observing conditions, lunation, '''
	# get nights on which this field was observed 

	# find blocks of nights where the moon was not highly illuminated

	flattened_path = '/data/tierras/flattened/'
	night_list = np.sort(glob(flattened_path+f'**/{field}/**'))
	calendar_dates = [i.split('/')[4] for i in night_list]
	calendar_datetimes = [Time(i[0:4]+'-'+i[4:6]+'-'+i[6:]) for i in calendar_dates]
	ut_datetimes = np.array([i+1 for i in calendar_datetimes])
	moon_illuminations = np.zeros(len(ut_datetimes))
	for i in range(len(ut_datetimes)):
		moon_illuminations[i] = astroplan.moon_illumination(ut_datetimes[i])
	breakpoint()
	return

def photometry(target, file_list, sources, ap_radii, an_in, an_out, type='fixed', centroid=False, bkg_type='1d'):
	date = file_list[0].parent.parent.parent.name 

	#file_list = file_list[181:]
	DARK_CURRENT = 0.00133 #e- pix^-1 s^-1. see Juliana's dissertation Table 4.1
	NONLINEAR_THRESHOLD = 40000. #ADU
	SATURATION_THRESHOLD = 55000. #ADU
	PLATE_SCALE = 0.43 #arcsec pix^-1, from Juliana's dissertation Table 1.1'

	#Set up arrays for doing photometry 

	#ARRAYS THAT CONTAIN DATA PERTAINING TO EACH FILE
	filenames = []
	mjd_utc = np.zeros(len(file_list),dtype='float')
	jd_utc = np.zeros(len(file_list),dtype='float')
	bjd_tdb = np.zeros(len(file_list),dtype='float')
	airmasses = np.zeros(len(file_list),dtype='float16')
	ccd_temps = np.zeros(len(file_list),dtype='float16')
	exp_times = np.zeros(len(file_list),dtype='float16')
	dome_temps = np.zeros(len(file_list),dtype='float16')
	focuses = np.zeros(len(file_list),dtype='float16')
	dome_humidities = np.zeros(len(file_list),dtype='float16')
	sec_temps = np.zeros(len(file_list),dtype='float16')
	ret_temps = np.zeros(len(file_list),dtype='float16')
	pri_temps = np.zeros(len(file_list),dtype='float16')
	rod_temps = np.zeros(len(file_list),dtype='float16')
	cab_temps = np.zeros(len(file_list),dtype='float16')
	inst_temps = np.zeros(len(file_list),dtype='float16')
	temps = np.zeros(len(file_list),dtype='float16')
	humidities = np.zeros(len(file_list),dtype='float16')
	dewpoints = np.zeros(len(file_list),dtype='float16')
	sky_temps = np.zeros(len(file_list),dtype='float16')
	pressures = np.zeros(len(file_list),dtype='float16')
	return_pressures = np.zeros(len(file_list),dtype='float16')
	supply_pressures = np.zeros(len(file_list),dtype='float16')
	hour_angles = np.zeros(len(file_list),dtype='float16')
	dome_azimuths = np.zeros(len(file_list),dtype='float16')
	wind_speeds = np.zeros(len(file_list),dtype='float16')
	wind_gusts = np.zeros(len(file_list),dtype='float16')
	wind_dirs = np.zeros(len(file_list),dtype='float16')

	loop_times = np.zeros(len(file_list),dtype='float16')
	#lunar_distance = np.zeros(len(file_list),dtype='float16')
	
	#ARRAYS THAT CONTAIN DATA PERTAINING TO EACH SOURCE IN EACH FILE
	source_x = np.zeros((len(sources),len(file_list)),dtype='float32')
	source_y = np.zeros((len(sources),len(file_list)),dtype='float32')
	source_sky_ADU = np.zeros((len(sources),len(file_list)),dtype='float32')
	source_x_fwhm_arcsec = np.zeros((len(sources),len(file_list)),dtype='float32')
	source_y_fwhm_arcsec = np.zeros((len(sources),len(file_list)),dtype='float32')
	source_theta_radians = np.zeros((len(sources),len(file_list)),dtype='float32')
	

	#ARRAYS THAT CONTAIN DATA PERTAININING TO EACH APERTURE RADIUS FOR EACH SOURCE FOR EACH FILE
	source_minus_sky_ADU = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	source_minus_sky_err_ADU = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	non_linear_flags = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='bool')
	saturated_flags = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='bool')
	
	source_radii = np.zeros((len(ap_radii),len(file_list)),dtype='float16')
	an_in_radii = np.zeros((len(ap_radii),len(file_list)),dtype='float16')
	an_out_radii = np.zeros((len(ap_radii),len(file_list)),dtype='float16')

	#Load in the stacked image of the field that was used for source identification. 
	#All images will be cross-correlated with this to determine aperture positions. 
	bpm = load_bad_pixel_mask()
	#Extra masking in case you need to fall back on astroalign, which does source detection. 
	bpm[0:1032, 1447:1464] = True
	bpm[1023:, 1788:1801]  = True
	#25-pixel mask on all edges
	bpm[:, 0:25+1] = True
	bpm[:,4096-1-25:] = True
	bpm[0:25+1,:] = True
	bpm[2048-1-25:,:] = True

	reference_image_hdu = fits.open('/data/tierras/targets/'+target+'/'+target+'_stacked_image.fits')[0] #TODO: should match image from target/reference csv file, and that should be loaded automatically.


	reference_image_header = reference_image_hdu.header
	reference_wcs = WCS(reference_image_header)
	
	reference_world_coordinates = np.array([sources['ra_tierras'],sources['dec_tierras']]).T

	reference_image_data = reference_image_hdu.data
	reference_image_data[np.where(bpm==1)] = np.nan

	#Background-subtract the data for figuring out shifts	
	try:
		bkg = sep.Background(reference_image_data)
	except:
		bkg = sep.Background(reference_image_data.byteswap().newbyteorder())

	reference_image_data -= bkg.back()
	
	# declare a circular footprint in case centroiding is performed
	# only data within a radius of x pixels around the expected source positions from WCS will be considered for centroiding
	circular_footprint_r = 3
	centroid_footprint = circular_footprint(circular_footprint_r)

	n_files = len(file_list)
	print(f'Doing fixed-radius circular aperture photometry for {len(sources)} sources in {n_files} images with aperture radii of {ap_radii} pixels, an inner annulus radius of {an_in} pixels, and an outer annulus radius of {an_out} pixels.\n')
	time.sleep(2)

	# fig1, ax1 = plt.subplots(1,3,figsize=(19,5),sharex=True,sharey=True)
	for i in range(n_files):
		if i > 0:
			loop_times[i-1]= time.time()-t_loop_start
			print(f'Avg loop time = {np.mean(loop_times[0:i]):.2f}s')
		t_loop_start = time.time()
		
		print(f'{i+1} of {n_files}')
		source_hdu = fits.open(file_list[i])[0]
		source_header = source_hdu.header
		source_data = source_hdu.data #TODO: Should we ignore BPM pixels?

		an_in_radii[:,i] = an_in
		an_out_radii[:,i] = an_out

		GAIN = source_header['GAIN'] #e- ADU^-1
		READ_NOISE = source_header['READNOIS'] #e-
		EXPTIME = source_header['EXPTIME']
		RA = source_header['RA']
		DEC = source_header['DEC']
		
		#SAVE ANCILLARY DATA
		filenames.append(file_list[i].name)
		mjd_utc[i] = source_header['MJD-OBS'] + (EXPTIME/2)/(24*60*60) #MJD-OBS is the modified julian date at the start of the exposure. Add on half the exposure time in days to get the time at mid-exposure. 
		jd_utc[i] = mjd_utc[i]+2400000.5 #Convert MJD_UTC to JD_UTC
		bjd_tdb[i] = jd_utc_to_bjd_tdb(jd_utc[i], RA, DEC)
		airmasses[i] = source_header['AIRMASS']
		ccd_temps[i] = source_header['CCDTEMP']
		exp_times[i] = source_header['EXPTIME']
		focuses[i] = source_header['FOCUS']
		#These keywords are sometimes missing
		try:
			dome_humidities[i] = source_header['DOMEHUMI']
			dome_temps[i] = source_header['DOMETEMP']
			sec_temps[i] = source_header['SECTEMP']
			rod_temps[i] = source_header['RODTEMP']
			cab_temps[i] = source_header['CABTEMP']
			inst_temps[i] = source_header['INSTTEMP']
			ret_temps[i] = source_header['RETTEMP']
			pri_temps[i] = source_header['PRITEMP']
			dewpoints[i] = source_header['DEWPOINT']
			temps[i] = source_header['TEMPERAT']
			humidities[i] = source_header['HUMIDITY']
			sky_temps[i] = source_header['SKYTEMP']
			pressures[i] = source_header['PRESSURE']
			return_pressures[i] = source_header['PSPRES1']
			supply_pressures[i] = source_header['PSPRES2']
			ha_str = source_header['HA']
			if ha_str[0] == '-':
				ha_decimal = int(ha_str.split(':')[0]) - int(ha_str.split(':')[1])/60 - float(ha_str.split(':')[2])/3600
			else:
				ha_decimal = int(ha_str.split(':')[0]) + int(ha_str.split(':')[1])/60 + float(ha_str.split(':')[2])/3600
			hour_angles[i] = ha_decimal
			dome_azimuths[i] = source_header['DOMEAZ']
			wind_speeds[i] = source_header['WINDSPD']
			wind_gusts[i] = source_header['WINDGUST']
			wind_dirs[i] = source_header['WINDDIR']
			
		except:
			dome_humidities[i] = np.nan
			dome_temps[i] = np.nan
			sec_temps[i] = np.nan
			rod_temps[i] = np.nan
			cab_temps[i] = np.nan
			inst_temps[i] = np.nan
			ret_temps[i] = np.nan
			pri_temps[i] = np.nan
			dewpoints[i] = np.nan
			temps[i] = np.nan
			humidities[i] = np.nan
			sky_temps[i] = np.nan
			pressures[i] = np.nan
			return_pressures[i] = np.nan
			supply_pressures[i] = np.nan
			hour_angles[i] = np.nan
			dome_azimuths[i] = np.nan
			wind_speeds[i] = np.nan
			wind_gusts[i] = np.nan
			wind_dirs[i] = np.nan

		#lunar_distance[i] = get_lunar_distance(RA, DEC, bjd_tdb[i]) #Commented out because this is slow and the information can be generated at a later point if necessary
		
		#Calculate expected scintillation noise in this image (see eqn. 2 in Juliana's SPIE paper) assuming all of the potential reference stars are uncorrelated 
		scintillation_rel = 1.5 * np.sqrt(1 + 1/(len(sources)-1)) * 0.09*(130)**(-2/3)*airmasses[i]**(7/4)*(2*EXPTIME)**(-1/2)*np.exp(-2306/8000)

		#UPDATE SOURCE POSITIONS
		#METHOD 1: WCS
		source_wcs = WCS(source_header)
		# transformed_pixel_coordinates = np.array([source_wcs.world_to_pixel(reference_world_coordinates[i]) for i in range(len(reference_world_coordinates))])
		transformed_pixel_coordinates = np.array(source_wcs.world_to_pixel_values(reference_world_coordinates))
		
		#Save transformed pixel coordinates of sources
		source_x[:,i] = transformed_pixel_coordinates[:,0]
		source_y[:,i] = transformed_pixel_coordinates[:,1]

		# create tuples of source positions in this frame
		source_positions = [(source_x[j,i], source_y[j,i]) for j in range(len(sources))]
		# ax.scatter(np.array([k[0] for k in source_positions]),np.array([k[1] for k in source_positions]), color='r', marker='x')

		if centroid:

			# t1 = time.time()
			# mask any pixels in the image above the non-linear threshold
			mask = np.zeros(np.shape(source_data), dtype='bool')
			mask[np.where(source_data>NONLINEAR_THRESHOLD)] = 1

			centroid_x, centroid_y = centroid_sources(source_data,source_x[:,i], source_y[:,i], centroid_func=centroid_com, footprint=centroid_footprint, mask=mask)
			
			# fig, ax = plot_image(source_data)
			# ax.scatter(centroid_x, centroid_y, color='b', marker='x')

			# make sure the centroids fall within the circular foot prints. If not, set to expected WCS position
			bad_centroids = np.where((abs(centroid_x-source_x[:,i]) > circular_footprint_r) | (abs(centroid_y-source_y[:,i]) > circular_footprint_r))[0]
			centroid_x[bad_centroids] = source_x[bad_centroids, i]
			centroid_y[bad_centroids] = source_y[bad_centroids, i]
			
			
			# update source positions
			source_x[:,i] = centroid_x 
			source_y[:,i] = centroid_y
			source_positions = [(source_x[j,i], source_y[j,i]) for j in range(len(sources))]
			# print(f'Centroids: {time.time()-t1}')
			
			# breakpoint()

		# Do photometry
		# Set up apertures
		if type == 'fixed':
			apertures = [CircularAperture(source_positions,r=ap_radii[k]) for k in range(len(ap_radii))]

		# check for non-linear/saturated pixels in the apertures 
		# just do in the smallest aperture for now  
		aperture_masks = apertures[0].to_mask(method='center')
		for j in range(len(apertures[0])):
			ap_cutout = aperture_masks[j].multiply(source_data)
			ap_pix_vals = ap_cutout[ap_cutout!=0]

			non_linear_flags[:,j,i] = int(np.sum(ap_pix_vals>NONLINEAR_THRESHOLD)>0)
			saturated_flags[:,j,i] = int(np.sum(ap_pix_vals>SATURATION_THRESHOLD)>0)
			
		#DO PHOTOMETRY AT UPDATED SOURCE POSITIONS FOR ALL SOURCES AND ALL APERTURES
		# t1 = time.time()
		if bkg_type == '1d':
			annuli = CircularAnnulus(source_positions, an_in, an_out)
			annulus_masks = annuli.to_mask(method='center')
			for j in range(len(annuli)):
				source_sky_ADU[j,i] = np.mean(sigmaclip(annulus_masks[j].get_values(source_data),2,2)[0])
			
			# print(f'Bkg: {time.time()-t1}')
		elif bkg_type == '2d':
				sigma_clip = SigmaClip(sigma=3.0)
				bkg_estimator = MedianBackground()
				bkg = Background2D(source_data, (32, 32), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, mask=mask)
				phot_table_2 = aperture_photometry(bkg.background, apertures)
				# Save the estimated per-pixel background by dividing the background in the largest aperture by the area of the largest aperture
				source_sky_ADU[:,i] = phot_table_2[f'aperture_sum_{len(ap_radii)-1}']/(np.pi*ap_radii[-1]**2)

		# do photometry on the *sources*
		phot_table = aperture_photometry(source_data, apertures)
		
		# Calculate sky-subtracted flux
		for k in range(len(ap_radii)):
			source_radii[k, i] = ap_radii[k]
			ap_area = apertures[k].area
			
			# for 1d background, subtract off the average bkg value measured in the annulus times the aperture area
			if bkg_type == '1d':
				source_minus_sky_ADU[k,:,i] = phot_table[f'aperture_sum_{k}']-source_sky_ADU[:,i]*ap_area

			# for 2d background, subtract off the aperture_sum in the aperture in the 2D background model
			elif bkg_type == '2d':
				source_minus_sky_ADU[k,:,i] = phot_table[f'aperture_sum_{k}'] - phot_table_2[f'aperture_sum_{k}']
				source_sky_ADU
				
				# norm1 = simple_norm(source_data, min_percent=1, max_percent=99)
				# ax1[0].imshow(source_data, origin='lower', norm=norm1)
				# ax1[0].plot(centroid_x, centroid_y, 'rx')

				# norm2 = simple_norm(bkg.background, min_percent=1, max_percent=99)
				# ax1[1].imshow(bkg.background, origin='lower', norm=norm2)

				# norm3 = simple_norm(source_data-bkg.background, min_percent=1, max_percent=99)
				# ax1[2].imshow(source_data-bkg.background, origin='lower', norm=norm3)
				# plt.tight_layout()
				# plt.pause(0.1)
				# breakpoint()
				# ax1[0].cla()
				# ax1[1].cla()
				# ax1[2].cla()


			# source_minus_sky_e[k,:,i] = source_minus_sky_ADU[k,:,i]*GAIN

			# convert scintillation noise from relative to absolutel by multiplying by source flux 
			sigma_scint = scintillation_rel * source_minus_sky_ADU[k,:,i] * GAIN
			
			# Calculate uncertainty
			# source_minus_sky_err_e = np.sqrt(source_minus_sky_ADU[k,:,i]*GAIN+ source_sky_ADU[:,i]*ap_area*GAIN + DARK_CURRENT*EXPTIME*ap_area + ap_area*READ_NOISE**2 + scintillation_abs_e**2)
			sigma_ccd = np.sqrt(source_minus_sky_ADU[k,:,i]*GAIN+ source_sky_ADU[:,i]*ap_area*GAIN + DARK_CURRENT*EXPTIME*ap_area + ap_area*READ_NOISE**2)

			# estimate total error by adding noise from ccd photon counting and scintillation in quadrature (e.g. equation 1 of Juliana's SPIE paper) 
			source_minus_sky_err_e = np.sqrt(sigma_ccd**2 + sigma_scint**2)
			source_minus_sky_err_ADU[k,:,i] = source_minus_sky_err_e / GAIN

		#Measure FWHM 
		k = 0
		t1 = time.time()
		for j in range(len(source_positions)):
			#g_2d_cutout = cutout[int(y_pos_cutout)-25:int(y_pos_cutout)+25,int(x_pos_cutout)-25:int(x_pos_cutout)+25]
			#t1 = time.time()

			#g_2d_cutout = copy.deepcopy(cutout)
			g_2d_cutout, cutout_pos = generate_square_cutout(source_data, source_positions[j], 40)
			
			bkg = np.mean(sigmaclip(g_2d_cutout,2,2)[0])

			xx2,yy2 = np.meshgrid(np.arange(g_2d_cutout.shape[1]),np.arange(g_2d_cutout.shape[0]))
			g_init = models.Gaussian2D(amplitude=g_2d_cutout[int(g_2d_cutout.shape[1]/2), int(g_2d_cutout.shape[0]/2)]-bkg,x_mean=cutout_pos[0],y_mean=cutout_pos[1], x_stddev=5, y_stddev=5)
			fit_g = fitting.LevMarLSQFitter()
			g = fit_g(g_init,xx2,yy2,g_2d_cutout-bkg)
			
			g.theta.value = g.theta.value % (2*np.pi) #Constrain from 0-2pi
			if g.y_stddev.value > g.x_stddev.value: 
				x_stddev_save = g.x_stddev.value
				y_stddev_save = g.y_stddev.value
				g.x_stddev = y_stddev_save
				g.y_stddev = x_stddev_save
				g.theta += np.pi/2

			x_stddev_pix = g.x_stddev.value
			y_stddev_pix = g.y_stddev.value 
			x_fwhm_pix = x_stddev_pix * 2*np.sqrt(2*np.log(2))
			y_fwhm_pix = y_stddev_pix * 2*np.sqrt(2*np.log(2))
			x_fwhm_arcsec = x_fwhm_pix * PLATE_SCALE
			y_fwhm_arcsec = y_fwhm_pix * PLATE_SCALE
			theta_rad = g.theta.value
			source_x_fwhm_arcsec[j,i] = x_fwhm_arcsec
			source_y_fwhm_arcsec[j,i] = y_fwhm_arcsec
			source_theta_radians[j,i] = theta_rad

			#print(time.time()-t1)

			# fig, ax = plt.subplots(1,2,figsize=(12,8),sharex=True,sharey=True)
			# norm = ImageNormalize(g_2d_cutout-bkg,interval=ZScaleInterval())
			# ax[0].imshow(g_2d_cutout-bkg,origin='lower',interpolation='none',norm=norm)
			# ax[1].imshow(g(xx2,yy2),origin='lower',interpolation='none',norm=norm)
			# plt.tight_layout()
			# breakpoint()
		# print(time.time()-t1)
		# breakpoint()

	for i in range(len(ap_radii)):
		output_dir = f'/home/ptamburo/tierras/pat_scripts/SAME/output/photometry/{date}'
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)

		
		output_path = Path(output_dir+f'/SAME_circular_{type}_ap_phot_{ap_radii[i]}.csv')
		
		output_list = []
		output_header = []
		
		output_list.append([f'{val}' for val in filenames])
		output_header.append('Filename')

		output_list.append([f'{val:.7f}' for val in mjd_utc])
		output_header.append('MJD UTC')
		output_list.append([f'{val:.7f}' for val in jd_utc])
		output_header.append('JD UTC')
		output_list.append([f'{val:.7f}' for val in bjd_tdb])
		output_header.append('BJD TDB')

		output_list.append([f'{val:.2f}' for val in source_radii[i]])
		output_header.append('Aperture Radius')
		output_list.append([f'{val:.2f}' for val in an_in_radii[i]])
		output_header.append('Inner Annulus Radius')
		output_list.append([f'{val:.2f}' for val in an_out_radii[i]])
		output_header.append('Outer Annulus Radius')

		output_list.append([f'{val:.2f}' for val in exp_times])
		output_header.append('Exposure Time')
		output_list.append([f'{val:.4f}' for val in airmasses])
		output_header.append('Airmass')
		output_list.append([f'{val:.1f}' for val in ccd_temps])
		output_header.append('CCD Temperature')
		output_list.append([f'{val:.2f}' for val in dome_temps])
		output_header.append('Dome Temperature')
		output_list.append([f'{val:.1f}' for val in focuses])
		output_header.append('Focus')
		output_list.append([f'{val:.2f}' for val in dome_humidities])
		output_header.append('Dome Humidity')
		output_list.append([f'{val:.1f}' for val in sec_temps])
		output_header.append('Sec Temperature')
		output_list.append([f'{val:.1f}' for val in ret_temps])
		output_header.append('Ret Temperature')
		output_list.append([f'{val:.1f}' for val in pri_temps])
		output_header.append('Pri Temperature')
		output_list.append([f'{val:.1f}' for val in rod_temps])
		output_header.append('Rod Temperature')
		output_list.append([f'{val:.2f}' for val in cab_temps])
		output_header.append('Cab Temperature')
		output_list.append([f'{val:.1f}' for val in inst_temps])
		output_header.append('Instrument Temperature')
		output_list.append([f'{val:.1f}' for val in temps])
		output_header.append('Temperature')
		output_list.append([f'{val:.1f}' for val in humidities])
		output_header.append('Humidity')
		output_list.append([f'{val:.2f}' for val in dewpoints])
		output_header.append('Dewpoint')
		output_list.append([f'{val:.1f}' for val in sky_temps])
		output_header.append('Sky Temperature')
		output_list.append([f'{val:.1f}' for val in pressures])
		output_header.append('Pressure')
		output_list.append([f'{val:.1f}' for val in return_pressures])
		output_header.append('Return Pressure')
		output_list.append([f'{val:.1f}' for val in supply_pressures])
		output_header.append('Supply Pressure')
		output_list.append([f'{val:.2f}' for val in hour_angles])
		output_header.append('Hour Angle')
		output_list.append([f'{val:.2f}' for val in dome_azimuths])
		output_header.append('Dome Azimuth')
		output_list.append([f'{val:.2f}' for val in wind_speeds])
		output_header.append('Wind Speed')
		output_list.append([f'{val:.2f}' for val in wind_gusts])
		output_header.append('Wind Gust')
		output_list.append([f'{val:.1f}' for val in wind_dirs])
		output_header.append('Wind Direction')

		#output_list.append([f'{val:.5f}' for val in lunar_distance])
		#output_header.append('Lunar Distance')

		for j in range(len(sources)):
			source_name = f'Source {j+1}'
			output_list.append([f'{val:.4f}' for val in source_x[j]])
			output_header.append(source_name+' X')
			output_list.append([f'{val:.4f}' for val in source_y[j]])
			output_header.append(source_name+' Y')
			output_list.append([f'{val:.7f}' for val in source_minus_sky_ADU[i,j]])
			output_header.append(source_name+' Source-Sky ADU')
			output_list.append([f'{val:.7f}' for val in source_minus_sky_err_ADU[i,j]])
			output_header.append(source_name+' Source-Sky Error ADU')

			output_list.append([f'{val:.7f}' for val in source_sky_ADU[j]])
			output_header.append(source_name+' Sky ADU')
			output_list.append([f'{val:.4f}' for val in source_x_fwhm_arcsec[j]])
			output_header.append(source_name+' X FWHM Arcsec')
			output_list.append([f'{val:.4f}' for val in source_y_fwhm_arcsec[j]])
			output_header.append(source_name+' Y FWHM Arcsec')
			output_list.append([f'{val:.4f}' for val in source_theta_radians[j]])
			output_header.append(source_name+' Theta Radians')


			output_list.append([f'{val:d}' for val in non_linear_flags[i,j]])
			output_header.append(source_name+' Non-Linear Flag')
			output_list.append([f'{val:d}' for val in saturated_flags[i,j]])
			output_header.append(source_name+' Saturated Flag')

		output_df = pd.DataFrame(np.transpose(output_list),columns=output_header)
		if not os.path.exists(output_path.parent.parent):
			os.mkdir(output_path.parent.parent)
			set_tierras_permissions(output_path.parent.parent)
		if not os.path.exists(output_path.parent):
			os.mkdir(output_path.parent)
			set_tierras_permissions(output_path.parent)
		output_df.to_csv(output_path,index=False)
		set_tierras_permissions(output_path)

	plt.close('all')
	return 		

def z_score_calculator(data_dict):
	''' 
		Caculate z scores of nightly medians for sources in a SAME data dictionary.

		Parameters:
		data_dict (dictionary): Dictionary containing corrected fluxes of sources generated with mearth_style.

		Returns:
		z_scores (array): array of z scores of nightly median fluxes for each source on each night
	'''
			
	flux_corr = data_dict['Corrected Flux']
	flux_corr_err = data_dict['Corrected Flux Error']
	zp_masks = data_dict['ZP Mask'].astype('bool')
	bjd_list = data_dict['BJD List']
	bjd = data_dict['BJD']

	n_stars = len(flux_corr)
	
	# median normalize the raw fluxes 
	norms2 = np.zeros(n_stars)
	for i in range(n_stars):
		flux_corr[i][~zp_masks[i]] = np.nan
		flux_corr_err[i][~zp_masks[i]] = np.nan 
		norms2[i] = np.nanmedian(flux_corr[i])

	flux_corr_norm = (flux_corr.T/norms2).T
	flux_corr_err_norm = (flux_corr_err.T/norms2).T
	
	# create arrays to store the median of the corrected flux for each source on each night 
	binned_flux = np.zeros((n_stars, len(bjd_list)))
	binned_flux_err = np.zeros_like(binned_flux)

	for ii in range(len(bjd_list)):
		try:
			use_inds = np.where((bjd >= bjd_list[ii][0])&(bjd <= bjd_list[ii][-1]))[0]
		except:
			continue
			
		binned_flux[:,ii] = np.nanmedian(flux_corr_norm[:,use_inds],axis=1)
		n_non_nan = sum(~np.isnan(flux_corr_norm[0,use_inds]))
		binned_flux_err[:,ii] = (1.2533/n_non_nan) *np.sqrt(np.nansum(flux_corr_err_norm[:,use_inds]**2, axis=1))	
		
	z_scores = (binned_flux-1)/binned_flux_err

	return z_scores, binned_flux, binned_flux_err

def summary_plots(data_dict, complist, cluster_ind, stars, all_stars, ap_radius, bkg_type='1d'):
	n_pix = np.pi*ap_radius**2
	GAIN = 5.9 # e- ADU^-1
	READ_NOISE = 15.5 #e- RMS
	DIAMETER = 130 #cm 
	ELEVATION = 2345 

	exptimes = data_dict['Exptime'] 
	flux = data_dict['Flux']
	flux_err = data_dict['Flux Error']
	flux_corr = data_dict['Corrected Flux']
	flux_corr_err = data_dict['Corrected Flux Error']
	sky = data_dict['Sky']
	zp_masks = data_dict['ZP Mask'].astype('bool')
	bjd_list = data_dict['BJD List']
	bjd = data_dict['BJD']
	fwhm_x = data_dict['FWHM X']
	fwhm_y = data_dict['FWHM Y']
	focus = data_dict['Focus']
	airmass = data_dict['Airmass']
	ha = data_dict['Hour Angle']
	x_pos = data_dict['X']
	y_pos = data_dict['Y']
	humidity = data_dict['Humidity']
	dome_humidity = data_dict['Dome Humidity']
	dewpoint = data_dict['Dewpoint']
	dome_temp = data_dict['Dome Temp']
	temp = data_dict['Temp']
	pri_temp = data_dict['Primary Temp']
	sec_temp = data_dict['Secondary Temp']
	inst_temp = data_dict['Instrument Temp']
	cab_temp = data_dict['Cabinet Temp']
	ret_temp = data_dict['Return Temp']
	rod_temp = data_dict['Rod Temp']
	ccd_temp = data_dict['CCD Temp']
	ret_pressure = data_dict['Return Pressure']
	supp_pressure = data_dict['Supply Pressure']
	sky_temp = data_dict['Sky Temp']
	dome_az = data_dict['Dome Azimuth']
	wind_spd = data_dict['Wind Speed']
	wind_gust = data_dict['Wind Gust']
	wind_dir = data_dict['Wind Direction']
	pressure = data_dict['Pressure']

	n_e = len(flux) - 1 # number of 'uncorrelated' reference stars used for each source, where those that are >20" apart are 'uncorrelated'; since this is < 5 tierras pixels, assume that all reference stars are uncorrelated
	
	# median normalize the raw fluxes 
	norms1 = np.zeros(len(flux))
	norms2 = np.zeros(len(flux))
	for i in range(len(flux)):
		flux[i][~zp_masks[i]] = np.nan 
		flux_err[i][~zp_masks[i]] = np.nan
		flux_corr[i][~zp_masks[i]] = np.nan
		flux_corr_err[i][~zp_masks[i]] = np.nan 
		norms1[i] = np.nanmedian(flux[i])
		norms2[i] = np.nanmedian(flux_corr[i])

	flux_norm = (flux.T/norms1).T
	flux_err_norm= (flux_err.T/norms1).T	
	flux_corr_norm = (flux_corr.T/norms2).T
	flux_corr_err_norm = (flux_corr_err.T/norms2).T
	
	# plt.figure()
	# x_sort = np.argsort(stars['X pix'])
	# cmap = plt.get_cmap('viridis')
	# for i in range(len(flux_norm)):
	# 	color = cmap(int(255*(stars['X pix'][x_sort[i]]-min(stars['X pix']))/(max(stars['X pix'])-min(stars['X pix']))))
	# 	bx, by, bye = tierras_binner(bjd, flux_norm[x_sort[i]], bin_mins=5)
	# 	# plt.plot(bjd, flux_norm[x_sort[i]], marker='.', ls='',color=color, alpha=0.5)
	# 	plt.errorbar(bx, by, bye, marker='o', ls='',mfc=color, mec='k', mew=1.5, zorder=4, ecolor=color)
	# breakpoint()

	# create arrays to store the median of the corrected flux for each source on each night 
	binned_flux = np.zeros((flux.shape[0], len(bjd_list)))
	binned_flux_err = np.zeros_like(binned_flux)

	plt.ioff()
	fig, ax_ = plt.subplots(18, len(bjd_list), figsize=(16,20), sharey='row', sharex=True)
	colors=  ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
	markers = ['.','*','s','v','^','p','P']
	random_offsets = np.random.uniform(-0.5, 0.5,flux.shape[0])
	for ii in range(len(bjd_list)):
		if len(bjd_list) == 1:
			ax = ax_
		else:
			ax = ax_[:,ii]
		try:
			use_inds = np.where((bjd >= bjd_list[ii][0])&(bjd <= bjd_list[ii][-1]))[0]
		except:
			continue
		plot_times = bjd[use_inds]
		if len(plot_times) == 0:
			continue

		ut_date = Time(plot_times[-1],format='jd',scale='tdb').iso.split(' ')[0]
		date_label = ut_date.split('-')[1].lstrip('0')+'/'+ut_date.split('-')[2].lstrip('0')+'/'+ut_date[2:4]
		plot_times -= plot_times[0]
		plot_times *= 24 
		mean_bjd = np.mean(plot_times)
		
		ax[0].set_title(date_label, rotation=0, fontsize=8)

		# plot normalized raw and normalized corrected fluxes
		marker_ind = 0
		for jj in range(flux.shape[0]):
			color = colors[(jj)%len(colors)]
			if ((jj) > 0) and ((jj)%len(colors)) == 0:
				marker_ind += 1 
			marker = markers[marker_ind]
	
			binned_flux[jj,ii] = np.nanmedian(flux_corr_norm[jj][use_inds])
			# binned_flux_err[jj,ii] = median_standard_error(flux_corr_norm[jj][use_inds]) # this approach uses boot strapping to estimate the error on the median

			# calculate sigma_rel, the expected relative uncertainty from counting photons with a CCD
			star_e = flux[jj][use_inds]*GAIN*exptimes[use_inds]
			sky_e = sky[jj][use_inds]*n_pix*GAIN*exptimes[use_inds]
			rn_e = n_pix*READ_NOISE**2 
			sigma_rel = np.sqrt(star_e + sky_e + rn_e)/star_e
			
			# calculate sigma_scint, expected relative uncertainty from scintillation 
			sigma_scint = 1.5*np.sqrt(1+1/n_e)*0.09*DIAMETER**(-2/3)*airmass[use_inds]**(7/4)*(2*exptimes[use_inds])**(-1/2)*np.exp(-ELEVATION/8000)

			# add these two uncertainties in quadrature to get the expected total uncertainty in each image 
			sigma_tot = np.sqrt(sigma_rel**2 + sigma_scint**2)

			error_on_mean_theory = np.sqrt(np.nansum(sigma_tot**2))/len(np.where(~np.isnan(star_e))[0])
			
			# binned_flux_err[jj,ii] = 1.2533*error_on_mean_theory # error on median should be ~25% larger than the theoretical error on the mean, according to the internet and Dave
			
			binned_flux_err[jj,ii] = np.sqrt(np.nansum(flux_corr_err_norm[0][use_inds]**2))/len(np.where(~np.isnan(star_e))[0]) * 1.2533 # alternative approach using the already-calculated theoretical errors, which should include photon noise (star and sky), read noise, dark current, scintillation, and noise from the zero-point correction

			ax[0].plot(plot_times, flux_norm[jj][use_inds], marker='.', ls='' ,color=color, alpha=0.5, mec='none', zorder=0)
			ax[1].plot(plot_times, flux_corr_norm[jj][use_inds], marker=marker, ls='' ,color=color, alpha=0.5, mec='none', zorder=0)
			if ii == len(bjd_list)-1:
				ax[1].plot(mean_bjd+random_offsets[jj], binned_flux[jj,ii], marker=marker,color=color, mec='k', mew=1.5, zorder=1,label=f'S{complist[jj]}', ls='', ms=20)
			else:
				ax[1].plot(mean_bjd+random_offsets[jj], binned_flux[jj,ii], marker=marker,color=color, mec='k', mew=1.5, zorder=1, ls='', ms=20)

			ax[3].plot(plot_times, sky[jj][use_inds], color=color, marker='.')
			ax[4].plot(plot_times, fwhm_x[jj][use_inds], color=color, marker='.')
			ax[5].plot(plot_times, fwhm_y[jj][use_inds], color=color, marker='.')
			ax[7].plot(plot_times, x_pos[jj][use_inds]-np.median(x_pos[jj]), label='X', color=color, marker='.')
			ax[8].plot(plot_times, y_pos[jj][use_inds]-np.median(y_pos[jj]), label='Y', color=color, marker='.')

		ax[0].grid(alpha=0.4)
		ax[1].grid(alpha=0.4)

		ax[2].plot(plot_times, airmass[use_inds], marker='.')
		ax_ha = ax[2].twinx()
		ax_ha.plot(plot_times, ha[use_inds], color='tab:orange', marker='.')
		ax_ha.set_ylim(min(ha)-0.25, max(ha)+0.25)

		ax[2].grid(alpha=0.4)


		
		# ax2 = ax[3].twinx()
		# ax2.plot(plot_times, focus[use_inds], label='Focus', color='tab:red')
		# ax2.set_ylim(min(focus)-1, max(focus)+1)

		ax[4].grid(alpha=0.4)
		ax[5].grid(alpha=0.4)

		ax[6].plot(plot_times, focus[use_inds], label='Focus', marker='.')
		ax[6].grid(alpha=0.4)

		
		ax[7].grid(alpha=0.4)
		ax[8].grid(alpha=0.4)


		ax[9].plot(plot_times, humidity[use_inds], label='Humidity', marker='.')
		ax[9].plot(plot_times, dome_humidity[use_inds], label='Dome humidity', marker='.')
		ax3 = ax[9].twinx()
		ax3.plot(plot_times, dewpoint[use_inds], label='Dewpoint', color='tab:red', marker='.')
		ax3.set_ylim(min(dewpoint)-2, max(dewpoint)+2)

		ax[9].grid(alpha=0.4)

		ax[10].plot(plot_times, dome_temp[use_inds], label='Dome', marker='.')
		ax[10].plot(plot_times, temp[use_inds], label='External', marker='.')
		ax[10].plot(plot_times, pri_temp[use_inds], label='Primary', marker='.')
		ax[10].plot(plot_times, sec_temp[use_inds], label='Secondary ', marker='.')
		ax[10].plot(plot_times, inst_temp[use_inds], label='Instrument', marker='.')
		ax[10].plot(plot_times, cab_temp[use_inds], label='Cabinet', marker='.')
		ax[10].plot(plot_times, ret_temp[use_inds], label='Return line', marker='.')
		ax[10].plot(plot_times, rod_temp[use_inds], label='Invar rod', marker='.')
		ax[10].grid(alpha=0.4)

		ax[11].plot(plot_times, ccd_temp[use_inds], label='CCD', marker='.')
		ax[11].grid(alpha=0.4)

		ax[12].plot(plot_times, ret_pressure[use_inds], label='Return', marker='.')
		ax[12].plot(plot_times, supp_pressure[use_inds], label='Supply', marker='.')
		ax[12].grid(alpha=0.4)

		ax[13].plot(plot_times, sky_temp[use_inds], label='Sky', marker='.')
		ax[13].grid(alpha=0.4)

		ax[14].plot(plot_times, dome_az[use_inds], label='Dome Azimuth', marker='.')
		ax[14].grid(alpha=0.4)


		ax[15].plot(plot_times, wind_spd[use_inds], label='Wind speed', marker='.')
		ax[15].plot(plot_times, wind_gust[use_inds], label='Wind gust', marker='.')
		ax[15].grid(alpha=0.4)

		ax[16].plot(plot_times, wind_dir[use_inds], label='Wind direction', marker='.')
		ax[16].grid(alpha=0.4)

		ax[17].plot(plot_times, pressure[use_inds], label='Ambient pressure', marker='.')
		ax[17].grid(alpha=0.4)

		if ii == len(bjd_list)-1:
			ax[1].legend(loc='center left', bbox_to_anchor=(1, 1), ncol=2)
			# ax2.set_ylabel('Focus', color='tab:red')
			ax_ha.set_ylabel('Hour Angle', color='tab:orange')
			ax[5].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
			#ax[5].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
			ax3.set_ylabel('Dewpoint', color='tab:red')
			ax[10].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
			ax[11].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
			ax[12].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
			ax[13].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
			ax[14].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
			ax[15].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
			ax[16].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
			ax[17].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
		else:
			# ax2.set_yticks([])
			ax_ha.set_yticks([])
			ax3.set_yticks([])
		if ii == 0:
			ax[0].set_ylabel('Raw Norm. Flux')
			ax[1].set_ylabel('Corr. Flux')
			ax[2].set_ylabel('Airmass')
			ax[3].set_ylabel('Sky (ADU/pix/s)')
			ax[4].set_ylabel('FWHM X')
			ax[5].set_ylabel('FWHM Y')
			ax[6].set_ylabel('Focus')
			ax[7].set_ylabel('X Pos.')
			ax[8].set_ylabel('Y Pos.')
			ax[9].set_ylabel('Humidity')
			ax[10].set_ylabel('Temp (C)')
			ax[11].set_ylabel('Temp (K)')
			ax[12].set_ylabel('Pressure (PSI)')
			ax[13].set_ylabel('Temp (C)')
			ax[14].set_ylabel('Dome az. (deg)')
			ax[15].set_ylabel('Wind speed (mph)')
			ax[16].set_ylabel('Wind dir (deg))')
			ax[17].set_ylabel('Pressure (hPa)')

	#ax2.tick_params(axis='y',labelcolor='tab:red')
	ax[4].legend()
	ax_ha.tick_params(axis='y', labelcolor='tab:orange')

	ax3.tick_params(axis='y',labelcolor='tab:red')
	ax[9].legend()
	

	ax[0].set_ylim(sigmaclip(flux_norm[0][~np.isnan(flux_norm[0])],5,5)[1:])
	ax[1].set_ylim(sigmaclip(flux_corr_norm[0][~np.isnan(flux_corr_norm[0])],5,5)[1:])
	ax[4].set_ylim(sigmaclip(fwhm_x[0],5,5)[1:])
	ax[5].set_ylim(sigmaclip(fwhm_y[0],5,5)[1:])

	plt.tight_layout()
	plt.subplots_adjust(left=0.04,right=0.8,hspace=0,wspace=0)
	output_dir = f'/home/ptamburo/tierras/pat_scripts/SAME/output/clusters/{cluster_ind}/'
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	plt.savefig(f'{output_dir}/summary.png',dpi=200)

	fig, ax = plt.subplots(1, 1, figsize=(12,5))
	offsets = np.linspace(-0.2, 0.2, len(binned_flux))
	marker_ind = 0
	z_scores = np.zeros_like(binned_flux)
	for ii in range(len(binned_flux)):
		color = colors[(ii)%len(colors)]
		if ((ii) > 0) and ((ii)%len(colors)) == 0:
			marker_ind += 1 
		marker = markers[marker_ind]
		ax.errorbar(np.arange(len(bjd_list))+offsets[ii],binned_flux[ii], binned_flux_err[ii], marker=marker,color=color, mec='k', mew=1., zorder=1,label=f'S{complist[ii]}', ms=10, ls='')
		z_scores[ii] = (binned_flux[ii] - 1)/binned_flux_err[ii]
	ax.set_ylabel('Night Median')
	ax.set_xlabel('Night Number')
	# plt.ylim(0.99, 1.015)
	ax.axhline(1,zorder=0,color='k', ls='--', alpha=0.8, lw=1)
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize=10)
	plt.tight_layout()
	plt.savefig(f'{output_dir}/offsets.png',dpi=200)


	# plot locations of stars on chip 
	# cluster_stars = stars[np.array(complist)]

	stacked_image_path = f'/data/tierras/targets/{target_field}/{target_field}_stacked_image.fits'
	hdu = fits.open(stacked_image_path)
	data = hdu[0].data
	header = hdu[0].header
	wcs = WCS(header)
	tierras_epoch = Time(header['TELDATE'],format='decimalyear')
	ra = header['RA']
	dec = header['DEC']
	# fig, ax = plot_image(data)
	fig, ax = plt.subplots(1,1,figsize=(13,8))
	ax.imshow(data, origin='lower', cmap='Greys_r', norm=simple_norm(data, min_percent=1,max_percent=99))
	plt.tight_layout()

	for i in range(len(flux)):
		ax.plot(stars['X pix'][i], stars['Y pix'][i], marker='x', mew=1.5, ms=10, color=colors[i%len(colors)])
		ax.text(stars['X pix'][i]+12, stars['Y pix'][i]+12, f'S{complist[i]}', fontsize=14, color=colors[i%len(colors)])
	
	plt.savefig(f'{output_dir}/field.png',dpi=200)


	# make a CMD 
	fig, ax = plt.subplots(1,1, figsize=(6,6))
	ax.plot(all_stars['bp_rp'],all_stars['gq_photogeo'], 'k.', alpha=0.4)
	for i in range(len(flux)):
		ax.plot(stars['bp_rp'][i], stars['gq_photogeo'][i], color=colors[i%len(colors)], marker='x', ms=12, mew=1.5)
	ax.invert_yaxis()
	ax.set_xlabel('B$_p$-R$_p$', fontsize=14)
	ax.set_ylabel('M$_{G}$', fontsize=14)
	ax.tick_params(labelsize=12)
	plt.tight_layout()
	plt.savefig(f'{output_dir}/cmd.png',dpi=200)
	plt.ion()
	
	# make a distribution of z scores 
	plt.figure(figsize=(8,6))
	weights = np.ones_like(np.ravel(z_scores))/len(np.ravel(z_scores))
	counts, bins, _ = plt.hist(np.ravel(z_scores), weights=weights, label=f'Observed ($\mu$={np.nanmean(np.ravel(z_scores)):.2f}, $\sigma$={np.nanstd(np.ravel(z_scores)):.2f})')
	z_score_theory = np.random.normal(0,1,100000)
	z_score_weights = np.ones_like(z_score_theory)/len(z_score_theory)
	theory_counts, theory_bins, _ = plt.hist(np.random.normal(0,1,100000), weights=z_score_weights, bins=bins, histtype='step', lw=2, label='Theory ($\mu$=0, $\sigma$=1)')
	plt.xlabel('Z score', fontsize=14)
	plt.ylabel('PDF', fontsize=14)
	ks_stat, ks_pval = kstest(np.sort(np.ravel(z_scores)), 'norm')
	plt.title(f'K-S test p-value: {ks_pval:.5f}')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig(f'{output_dir}/zscores.png',dpi=200)

	plt.close('all')

	plt.ion()	
	return fig, ax, binned_flux, binned_flux_err

def mearth_style_pat_weighted_flux(data_dict):
	""" Use the comparison stars to derive a frame-by-frame zero-point magnitude. Also filter and mask bad cadences """
	""" it's called "mearth_style" because it's inspired by the mearth pipeline """
	""" this version works with fluxes bc I hate the magnitude system with a burning passion"""
	
	bjds = data_dict['BJD']
	flux = data_dict['Flux']
	flux_err = data_dict['Flux Error']

	# mask any cadences where the flux is negative for any of the sources 
	mask = np.ones_like(bjds, dtype='bool')  # initialize a bad data mask
	for i in range(len(flux)):
		mask[np.where(flux[i] <= 0)[0]] = 0  

	flux_corr_save = np.zeros_like(flux)
	flux_err_corr_save = np.zeros_like(flux)
	mask_save = np.zeros_like(flux)
	weights_save = np.zeros((len(flux),len(flux)-1))

	# loop over each star, calculate its zero-point correction using the other stars
	for i in range(len(flux)):
		# target_source_id = cluster_ids[i] # this represents the ID of the "target" *in the photometry files
		regressor_inds = [j for j in np.arange(len(flux)) if i != j] # get the indices of the stars to use as the zero point calibrators; these represent the indices of the calibrators *in the data_dict arrays*
		# regressor_source_ids = cluster_ids[regressor_inds] # these represent the IDs of the calibrators *in the photometry files*  

		# grab target and source fluxes and apply initial mask 
		target_flux = data_dict['Flux'][i]
		target_flux_err = data_dict['Flux Error'][i]
		regressors = data_dict['Flux'][regressor_inds]
		regressors_err = data_dict['Flux Error'][regressor_inds]

		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		for j in range(len(regressors)):
			regressors[j][~mask] = np.nan 

		tot_regressor = np.sum(regressors, axis=0)  # the total regressor flux at each time point = sum of comp star fluxes in each exposure
		tot_regressor[~mask] = np.nan

		# identify cadences with "low" flux by looking at normalized summed reference star fluxes
		zp0s = tot_regressor/np.nanmedian(tot_regressor) 	
		mask = np.ones_like(zp0s, dtype='bool')  # initialize another bad data mask
		mask[np.where(zp0s < 0.8)[0]] = 0  # if regressor flux is decremented by 20% or more, this cadence is bad
		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		for j in range(len(regressors)):
			regressors[j][~mask] = np.nan 

		# repeat the cs estimate now that we've masked out the bad cadences
		# phot_regressor = np.nanpercentile(regressors, 90, axis=1)  # estimate photometric flux level for each star
		norms = np.nanmedian(regressors, axis=1)
		regressors_norm = regressors / norms[:, None]
		regressors_err_norm = regressors_err / norms[:, None]

		# mask out any exposures where any reference star is significantly discrepant 
		mask = np.ones_like(target_flux, dtype='bool')
		for j in range(len(regressors_norm)):
			v, l, h = sigmaclip(regressors_norm[j][~np.isnan(regressors_norm[j])])
			mask[np.where((regressors_norm[j] < l) | (regressors_norm[j] > h))[0]] = 0

		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		for j in range(len(regressors)):
			regressors[j][~mask] = np.nan 
			regressors_err[j][~mask] = np.nan
			regressors_norm[j][~mask] = np.nan
			regressors_err_norm[j][~mask] = np.nan

		# now calculate the weights for each regressor
		# give all stars equal weights at first
		weights_init = np.ones(len(regressors))/len(regressors)
		
		# calculate the initial zero point correction and its uncertaintiy 
		zp = np.matmul(weights_init, regressors_norm) 
		zp_err = np.sqrt(np.matmul(weights_init**2, regressors_err_norm**2)) # this represents the error on the weighted mean of all the regressors at each point 
		
		zp_init = copy.deepcopy(zp)
		zp_err_init = copy.deepcopy(zp_err)
		# plot the normalized regressor fluxes and the initial zero-point correction
		# for j in range(len(regressors)):
		# 	plt.plot(regressors_norm[j])
		# plt.errorbar(np.arange(len(zp)), zp, zp_err, color='k', zorder=4)

		# do a 'crude' weighting loop to figure out which regressors, if any, should be totally discarded	
		delta_weights = np.zeros(len(regressors))+999 # initialize
		threshold = 1e-4 # delta_weights must converge to this value for the loop to stop
		weights_old = weights_init
		full_ref_inds = np.arange(len(regressors))
		while len(np.where(delta_weights>threshold)[0]) > 0:
			stddevs = np.zeros(len(regressors))		
			
			# loop over each regressor
			for jj in range(len(regressors)):
				# make its zeropoint correction using the flux of all the *other* regressors
				use_inds = np.delete(full_ref_inds, jj)

				# re-normalize the weights to sum to one
				weights_wo_jj = weights_old[use_inds]
				weights_wo_jj /= np.nansum(weights_wo_jj)
				
				# create a zeropoint correction using those weights 
				zp_wo_jj = np.matmul(weights_wo_jj, regressors_norm[use_inds])

				# correct the source's flux and re-normalize
				corr_jj = regressors_norm[jj] / zp_wo_jj
				corr_jj /= np.nanmedian(corr_jj)

				# record the standard deviation of the corrected flux
				stddevs[jj] = np.nanstd(corr_jj)

			# update the weights using the measured standard deviations
			weights_new = 1/stddevs**2
			weights_new /= np.nansum(weights_new)
			delta_weights = abs(weights_new-weights_old)
			weights_old = weights_new

		weights = weights_new

		# determine if any references should be totally thrown out based on the ratio of their measured/expected noise
		regressors_err_norm = (regressors_err.T / np.nanmedian(regressors,axis=1)).T
		noise_ratios = stddevs / np.nanmedian(regressors_err_norm)      

		# the noise ratio threshold will depend on how many bad/variable reference stars were used in the ALC
		# sigmaclip the noise ratios and set the upper limit to the n-sigma upper bound 
		v, l, h = sigmaclip(noise_ratios, 2, 2)

		weights[np.where(noise_ratios>h)[0]] = 0
		weights /= sum(weights)
		
		if len(np.where(weights == 0)[0]) > 0:
			# now repeat the weighting loop with the bad refs removed 
			delta_weights = np.zeros(len(regressors))+999 # initialize
			threshold = 1e-6 # delta_weights must converge to this value for the loop to stop
			weights_old = weights
			full_ref_inds = np.arange(len(regressors))
			count = 0
			while len(np.where(delta_weights>threshold)[0]) > 0:
				stddevs = np.zeros(len(regressors))

				for jj in range(len(regressors)):
					if weights_old[jj] == 0:
						continue
					use_inds = np.delete(full_ref_inds, jj)
					weights_wo_jj = weights_old[use_inds]
					weights_wo_jj /= np.nansum(weights_wo_jj)
					zp_wo_jj = np.matmul(weights_wo_jj, regressors_norm[use_inds])	
					
					corr_jj = regressors[jj] / zp_wo_jj
					corr_jj /= np.nanmean(corr_jj)
					stddevs[jj] = np.nanstd(corr_jj)

				weights_new = 1/(stddevs**2)
				weights_new /= np.sum(weights_new[~np.isinf(weights_new)])
				weights_new[np.isinf(weights_new)] = 0
				delta_weights = abs(weights_new-weights_old)
				weights_old = weights_new
				count += 1

		weights = weights_new

		# calculate the zero-point correction
		zp = np.matmul(weights, regressors_norm)
		zp_err = np.sqrt(np.matmul(weights**2, regressors_err_norm**2)) # this represents the error on the weighted mean of all the regressors at each point 
		
		flux_corr = target_flux / zp  #cs, adjust the flux based on the calculated zero points
		err_corr = np.sqrt((target_flux_err/zp)**2 + (target_flux*zp_err/(zp**2))**2)
		mask_save[i] = ~np.isnan(flux_corr)
		flux_corr_save[i] = flux_corr
		flux_err_corr_save[i] = err_corr
		weights_save[i] = weights

	output_dict = copy.deepcopy(data_dict)
	output_dict['ZP Mask'] = mask_save
	output_dict['Corrected Flux'] = flux_corr_save
	output_dict['Corrected Flux Error'] = flux_err_corr_save
	output_dict['Weights'] = weights_save

	return output_dict

def mearth_style_pat_weighted(data_dict):
	""" Use the comparison stars to derive a frame-by-frame zero-point magnitude. Also filter and mask bad cadences """
	""" it's called "mearth_style" because it's inspired by the mearth pipeline """
	# cluster_ids = np.array(cluster_ids)

	bjds = data_dict['BJD']
	flux = data_dict['Flux']
	flux_err = data_dict['Flux Error']

	# mask any cadences where the flux is negative for any of the sources 
	mask = np.ones_like(bjds, dtype='bool')  # initialize a bad data mask
	for i in range(len(flux)):
		mask[np.where(flux[i] <= 0)[0]] = 0  

	flux_corr_save = np.zeros_like(flux)
	flux_err_corr_save = np.zeros_like(flux)
	mask_save = np.zeros_like(flux)
	weights_save = np.zeros((len(flux),len(flux)-1))

	# loop over each star, calculate its zero-point correction using the other stars
	for i in range(len(flux)):
		# target_source_id = cluster_ids[i] # this represents the ID of the "target" *in the photometry files
		regressor_inds = [j for j in np.arange(len(flux)) if i != j] # get the indices of the stars to use as the zero point calibrators; these represent the indices of the calibrators *in the data_dict arrays*
		# regressor_source_ids = cluster_ids[regressor_inds] # these represent the IDs of the calibrators *in the photometry files*  

		# grab target and source fluxes and apply initial mask 
		target_flux = data_dict['Flux'][i]
		target_flux_err = data_dict['Flux Error'][i]
		regressors = data_dict['Flux'][regressor_inds]
		regressors_err = data_dict['Flux Error'][regressor_inds]

		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		for j in range(len(regressors)):
			regressors[j][~mask] = np.nan 

		tot_regressor = np.sum(regressors, axis=0)  # the total regressor flux at each time point = sum of comp star fluxes in each exposure
		tot_regressor[~mask] = np.nan

		c0s = -2.5*np.log10(np.nanpercentile(tot_regressor, 90)/tot_regressor)  # initial guess of magnitude zero points
		
		mask = np.ones_like(c0s, dtype='bool')  # initialize another bad data mask
		mask[np.where(c0s < -0.24)[0]] = 0  # if regressor flux is decremented by 20% or more, this cadence is bad

		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		for j in range(len(regressors)):
			regressors[j][~mask] = np.nan 

		# repeat the cs estimate now that we've masked out the bad cadences
		phot_regressor = np.nanpercentile(regressors, 90, axis=1)  # estimate photometric flux level for each star
		cs = -2.5*np.log10(phot_regressor[:,None]/regressors)  # estimate c for each star
		c_noise = np.nanstd(cs, axis=0)  # estimate the error in c
		c_unc = (np.nanpercentile(cs, 84, axis=0) - np.nanpercentile(cs, 16, axis=0)) / 2.  # error estimate that ignores outliers

		''' c_unc will overestimate error introduced by zero-point offset because it is correlated. Attempt to correct
		for this by only considering the additional error compared to the cadence where c_unc is minimized '''
		c_unc_best = np.nanmin(c_unc)
		c_unc = np.sqrt(c_unc**2 - c_unc_best**2)

		# Initialize weights using average fluxes of the regressors
		# weights_init = np.nanmean(regressors, axis=1)
		# weights_init /= np.nansum(weights_init) # Normalize weights to sum to 1

		# give all stars equal weights at first
		weights_init = np.ones(len(regressors))/len(regressors)

		cs = np.matmul(weights_init, cs) # Take the *weighted mean* across all regressors

		# one more bad data mask: don't trust cadences where the regressors have big discrepancies
		mask = np.ones_like(target_flux, dtype='bool')
		mask[np.where(c_noise > 3*np.median(c_noise))[0]] = 0

		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		for j in range(len(regressors)):
			regressors[j][~mask] = np.nan 
		c_unc[~mask] = np.nan

		cs_original = cs
		delta_weights = np.zeros(len(regressors))+999 # initialize
		threshold = 1e-4 # delta_weights must converge to this value for the loop to stop
		weights_old = weights_init
		full_ref_inds = np.arange(len(regressors))
		while len(np.where(delta_weights>threshold)[0]) > 0:
			stddevs = np.zeros(len(regressors))
			cs = -2.5*np.log10(phot_regressor[:,None]/regressors)

			for jj in range(len(regressors)):
				use_inds = np.delete(full_ref_inds, jj)
				weights_wo_jj = weights_old[use_inds]
				weights_wo_jj /= np.nansum(weights_wo_jj)
				cs_wo_jj = np.matmul(weights_wo_jj, cs[use_inds])
				corr_jj = regressors[jj] * 10**(-cs_wo_jj/2.5)
				# reg_flux, intercept, coeffs, ancillary_dict_return = regression(corr_jj, ancillary_dict, pval_threshold=1e-3, verbose=False)
				corr_jj /= np.nanmedian(corr_jj)
				stddevs[jj] = np.nanstd(corr_jj)

			weights_new = 1/stddevs**2
			weights_new /= np.nansum(weights_new)
			delta_weights = abs(weights_new-weights_old)
			weights_old = weights_new

		weights = weights_new

		# determine if any references should be totally thrown out based on the ratio of their measured/expected noise
		regressors_err_norm = (regressors_err.T / np.nanmedian(regressors,axis=1)).T
		noise_ratios = stddevs / np.nanmedian(regressors_err_norm)      

		# the noise ratio threshold will depend on how many bad/variable reference stars were used in the ALC
		# sigmaclip the noise ratios and set the upper limit to the n-sigma upper bound 
		v, l, h = sigmaclip(noise_ratios, 2, 2)
		weights[np.where(noise_ratios>h)[0]] = 0
		weights /= sum(weights)
		
		if len(np.where(weights == 0)[0]) > 0:
			# now repeat the weighting loop with the bad refs removed 
			delta_weights = np.zeros(len(regressors))+999 # initialize
			threshold = 1e-6 # delta_weights must converge to this value for the loop to stop
			weights_old = weights
			full_ref_inds = np.arange(len(regressors))
			count = 0
			while len(np.where(delta_weights>threshold)[0]) > 0:
				stddevs = np.zeros(len(regressors))
				cs = -2.5*np.log10(phot_regressor[:,None]/regressors)

				for jj in range(len(regressors)):
					if weights_old[jj] == 0:
						continue
					use_inds = np.delete(full_ref_inds, jj)
					weights_wo_jj = weights_old[use_inds]
					weights_wo_jj /= np.nansum(weights_wo_jj)
					cs_wo_jj = np.matmul(weights_wo_jj, cs[use_inds])
					corr_jj = regressors[jj] * 10**(-cs_wo_jj/2.5)
					corr_jj /= np.nanmean(corr_jj)
					stddevs[jj] = np.nanstd(corr_jj)
				weights_new = 1/(stddevs**2)
				weights_new /= np.sum(weights_new[~np.isinf(weights_new)])
				weights_new[np.isinf(weights_new)] = 0
				delta_weights = abs(weights_new-weights_old)
				weights_old = weights_new
				count += 1

		weights = weights_new

		# calculate the zero-point correction
		cs = -2.5*np.log10(phot_regressor[:,None]/regressors)

		# recalculate c_unc accounting for the derived weights
		weighted_mean = np.average(cs, weights=weights, axis=0)
		weighted_variance = np.average((cs-weighted_mean)**2, weights=weights, axis=0)
		c_unc = np.sqrt(weighted_variance)
		''' c_unc will overestimate error introduced by zero-point offset because it is correlated. Attempt to correct
		for this by only considering the additional error compared to the cadence where c_unc is minimized '''
		c_unc_best = np.nanmin(c_unc)
		c_unc = np.sqrt(c_unc**2 - c_unc_best**2)

		norm = np.nanmedian(regressors, axis=1)
		reg_norm = (regressors.T/norm).T
		reg_err_norm = (regressors_err.T/norm).T
		# for j in range(len(cs)):
		# 	plt.plot(10**(-cs[j]/2.5), marker='.')
		# 	plt.plot(reg_norm)
		# breakpoint()
		cs = np.matmul(weights, cs)
		
		corrected_regressors = regressors * 10**(-cs/2.5)
		
		# flux_original = copy.deepcopy(flux)
		err_corr = 10**(cs/(-2.5)) * np.sqrt(target_flux_err**2 + (c_unc*target_flux*np.log(10)/(-2.5))**2)  # propagate error

		flux_corr = target_flux*10**(cs/(-2.5))  #cs, adjust the flux based on the calculated zero points
		# breakpoint()	
		mask_save[i] = ~np.isnan(flux_corr)
		flux_corr_save[i] = flux_corr
		flux_err_corr_save[i] = err_corr
		weights_save[i] = weights

	output_dict = copy.deepcopy(data_dict)
	output_dict['ZP Mask'] = mask_save
	output_dict['Corrected Flux'] = flux_corr_save
	output_dict['Corrected Flux Error'] = flux_err_corr_save
	output_dict['Weights'] = weights_save

	return output_dict

def return_dataframe_onedate(mainpath,targetname,obsdate): #JGM MODIFIED NOV 9/2023
	datepath = os.path.join(mainpath,obsdate,targetname)
	optimal_lc_fname = os.path.join(datepath,'optimal_lc.txt') #returns only optimal aperture lc
	try:
		optimal_lc_csv = open(optimal_lc_fname).read().rstrip('\n')
		df = pd.read_csv(optimal_lc_csv)
	except FileNotFoundError:
		print ("No photometric extraction for {} on {}".format(targetname,obsdate))
		return None
	return df, optimal_lc_csv

def return_dataframe_onedate_forapradius(mainpath,obsdate,ap_radius='optimal', bkg_type='1d'): #JGM MODIFIED JAN3/2024: return dataframe for a user-defined aperture.
		if ap_radius == 'optimal': # simply use optimal radius according to ap_phot. Needs work. 
				df,lc_fname = return_dataframe_onedate(mainpath,obsdate)
				print('Optimal ap radius: ',lc_fname.split('_')[-1].split('.csv')[0])
				return df,lc_fname
		else:
				datepath = os.path.join(mainpath,obsdate)
				lc_fname = os.path.join(datepath,'SAME_circular_fixed_ap_phot_{}.csv'.format(str(ap_radius)))
				print (ap_radius)
				try:
						df = pd.read_csv(lc_fname)
				except FileNotFoundError:
						print ("No photometric extraction for {} with aperture radius of {} pixels on {}".format(targetname,ap_radius,obsdate))
						return None
				return df, lc_fname

def make_global_lists(cluster_ids, mainpath, night_list=[], ap_radius='optimal', bkg_type='1d'):
	# arrays to hold the full dataset
	full_bjd = []
	# full_flux = []
	# full_err = []
	# full_flux_div_expt = [] # sometimes data from one star has different exposure times in a given night or between nights
	# full_err_div_expt = []
	
	full_flux = None
	full_err = None 
	full_flux_div_expt = None 
	full_err_div_expt = None 
	# full_reg = None
	# full_reg_err = None

	full_relflux = []
	full_exptime = []
	full_sky = None
	full_x = None
	full_y = None
	full_airmass = []
	full_fwhm_x = None
	full_fwhm_y = None
	full_humidity = []
	full_dome_humidity = []
	full_ccd_temp = []
	full_dome_temp = []
	full_focus = []
	full_sec_temp = []
	full_ret_temp = []
	full_pri_temp = []
	full_rod_temp = []
	full_cab_temp = []
	full_inst_temp = []
	full_temp = []
	full_dewpoint = []
	full_sky_temp = []
	full_pressure = []
	full_ret_pressure = []
	full_supp_pressure = []
	full_ha = []
	full_dome_az = []
	full_wind_spd = []
	full_wind_gust = []
	full_wind_dir = []

	#full_corr_relflux = [] 

	# array to hold individual nights
	bjd_save = []
	if len(night_list) > 0:
		lcdatelist = night_list
		lcfolderlist = [mainpath+'/'+i for i in lcdatelist]
	else:
		lcfolderlist = np.sort(glob(mainpath+f'/2**'))
		lcdatelist = [lcfolderlist[ind].split("/")[-1] for ind in range(len(lcfolderlist))] 
	for ii,lcfolder in enumerate(lcfolderlist):
		print("Processing", lcdatelist[ii])

		# # if date excluded, skip
		# if np.any(exclude_dates == lcdatelist[ii]):
		# 	print ("{} :  Excluded".format(lcdatelist[ii]))
		# 	continue

		# read the .csv file
		try:
			df, optimal_lc = return_dataframe_onedate_forapradius(mainpath,lcdatelist[ii], ap_radius, bkg_type=bkg_type)
		except TypeError:
			continue

		bjds = df['BJD TDB'].to_numpy()
		flux = np.zeros((len(cluster_ids),len(bjds)))
		err = np.zeros_like(flux)
		x = np.zeros_like(flux)
		y = np.zeros_like(flux)
		sky = np.zeros_like(flux)
		fwhm_x = np.zeros_like(flux)
		fwhm_y = np.zeros_like(flux)
		
		expt = df['Exposure Time']
		for i in range(len(cluster_ids)):
			flux[i] = df[f'Source {cluster_ids[i]+1} Source-Sky ADU'] / expt
			err[i] = df[f'Source {cluster_ids[i]+1} Source-Sky Error ADU'] / expt
			x[i] = df[f'Source {cluster_ids[i]+1} X']
			y[i] = df[f'Source {cluster_ids[i]+1} Y']
			sky[i] = df[f'Source {cluster_ids[i]+1} Sky ADU'] / expt
			fwhm_x[i] = df[f'Source {cluster_ids[i]+1} X FWHM Arcsec']
			fwhm_y[i] = df[f'Source {cluster_ids[i]+1} Y FWHM Arcsec']

		airmass = df['Airmass']
		humidity = df['Humidity']
		dome_humidity = df['Dome Humidity']
		ccd_temp = df['CCD Temperature']
		dome_temp = df['Dome Temperature']
		focus = df['Focus']
		sec_temp = df['Sec Temperature']
		pri_temp = df['Pri Temperature']
		ret_temp = df['Ret Temperature']
		rod_temp = df['Rod Temperature']
		cab_temp = df['Cab Temperature']
		inst_temp  = df['Instrument Temperature']
		temp = df['Temperature']
		dewpoint = df['Dewpoint']
		sky_temp = df['Sky Temperature']
		pressure = df['Pressure']
		ret_pressure = df['Return Pressure']
		supp_pressure = df['Supply Pressure']
		ha = df['Hour Angle']
		dome_az = df['Dome Azimuth']
		wind_spd = df['Wind Speed']
		wind_gust = df['Wind Gust']
		wind_dir = df['Wind Direction']

		# # get the comparison fluxes.
		# comps = {}
		# comps_err = {}
		# fwhm_x = {}
		# fwhm_y = {}
		# fwhm_x[0] = df[f'Source {source_ind+1} X FWHM Arcsec']
		# fwhm_y[0] = df[f'Source {source_ind+1} Y FWHM Arcsec']
		# for comp_num in complist:
		# 	try:
		# 		fwhm_x[comp_num] = df['Source '+str(comp_num+1)+' X FWHM Arcsec']
		# 		fwhm_y[comp_num] = df['Source '+str(comp_num+1)+' Y FWHM Arcsec']
		# 		comps[comp_num] = df['Source '+str(comp_num+1)+' Source-Sky ADU'] / expt  # divide by exposure time since it can vary between nights
		# 		comps_err[comp_num] = df['Source '+str(comp_num+1)+' Source-Sky Error ADU'] / expt
		# 	except:
		# 		print("Error with comp", str(comp_num+1))
		# 		continue

		# # make a list of all the comps
		# regressors = []
		# regressors_err = []
		# fwhm_x_ = []
		# fwhm_y_ = []
		# fwhm_x_.append(fwhm_x[0])
		# fwhm_y_.append(fwhm_y[0])
		# for key in comps.keys():
		# 	regressors.append(comps[key])
		# 	regressors_err.append(comps_err[key])
		# 	fwhm_x_.append(fwhm_x[key])
		# 	fwhm_y_.append(fwhm_y[key])


		# regressors = np.array(regressors)
		# regressors_err = np.array(regressors_err)

		# fwhm_x = np.array(fwhm_x_)
		# fwhm_y = np.array(fwhm_y_)

		# add this night of data to the full data set
		full_bjd.extend(bjds)
		# full_flux.extend(flux)
		# full_err.extend(err)
		# full_flux_div_expt.extend(flux/expt)
		# full_err_div_expt.extend(err/expt)		
		bjd_save.append(bjds)

		# full_relflux.extend(relflux)
		full_exptime.extend(expt)
		# full_sky.extend(sky/expt)
		# full_x.extend(x)
		# full_y.extend(y)
		
		full_airmass.extend(airmass)
		full_humidity.extend(humidity)
		full_dome_humidity.extend(dome_humidity)
		full_ccd_temp.extend(ccd_temp)
		full_dome_temp.extend(dome_temp)
		full_focus.extend(focus)
		full_sec_temp.extend(sec_temp)
		full_pri_temp.extend(pri_temp)
		full_ret_temp.extend(ret_temp)
		full_rod_temp.extend(rod_temp)
		full_cab_temp.extend(cab_temp)
		full_inst_temp.extend(inst_temp)
		full_temp.extend(temp)
		full_dewpoint.extend(dewpoint)
		full_sky_temp.extend(sky_temp)
		full_pressure.extend(pressure)
		full_ret_pressure.extend(ret_pressure)
		full_supp_pressure.extend(supp_pressure)
		full_ha.extend(ha)
		full_dome_az.extend(dome_az)
		full_wind_spd.extend(wind_spd)
		full_wind_gust.extend(wind_gust)
		full_wind_dir.extend(wind_dir)

		if full_flux is None:
			full_flux = flux
			full_flux_err = err
			full_x_pos = x
			full_y_pos = y 
			full_fwhm_x = fwhm_x
			full_fwhm_y = fwhm_y
			full_sky = sky 
		else:
			full_flux = np.concatenate((full_flux, flux), axis=1) 
			full_flux_err = np.concatenate((full_flux_err, err), axis=1)
			full_x_pos = np.concatenate((full_x_pos, x), axis=1)
			full_y_pos = np.concatenate((full_y_pos, y), axis=1)
			full_fwhm_x = np.concatenate((full_fwhm_x, fwhm_x),axis=1)
			full_fwhm_y = np.concatenate((full_fwhm_y, fwhm_y),axis=1)
			full_sky = np.concatenate((full_sky, sky),axis=1)

	# convert from lists to arrays
	full_bjd = np.array(full_bjd)
	full_flux = np.array(full_flux)
	full_flux_err = np.array(full_flux_err)
	#full_reg_err = np.array(full_reg_err)
	# full_flux_div_expt = np.array(full_flux_div_expt)
	# full_err_div_expt =np.array(full_err_div_expt)
	# full_relflux = np.array(full_relflux)
	full_exptime = np.array(full_exptime)
	full_sky = np.array(full_sky)
	full_x = np.array(full_x_pos)
	full_y = np.array(full_y_pos)
	full_airmass = np.array(full_airmass)
	full_fwhm_x = np.array(full_fwhm_x)
	full_fwhm_y = np.array(full_fwhm_y)
	full_humidity = np.array(full_humidity)
	full_dome_humidity = np.array(full_dome_humidity)
	full_ccd_temp = np.array(full_ccd_temp)
	full_dome_temp = np.array(full_dome_temp)
	full_focus = np.array(full_focus)
	full_sec_temp = np.array(full_sec_temp)
	full_pri_temp = np.array(full_pri_temp)
	full_ret_temp = np.array(full_ret_temp)
	full_rod_temp = np.array(full_rod_temp)
	full_cab_temp = np.array(full_cab_temp)
	full_inst_temp = np.array(full_inst_temp)
	full_temp = np.array(full_temp)
	full_dewpoint = np.array(full_dewpoint)
	full_sky_temp = np.array(full_sky_temp)
	full_pressure = np.array(full_pressure)
	full_ret_pressure  = np.array(full_ret_pressure)
	full_supp_pressure  = np.array(full_supp_pressure)
	full_ha  = np.array(full_ha)
	full_dome_az  = np.array(full_dome_az)
	full_wind_spd  = np.array(full_wind_spd)
	full_wind_gust  = np.array(full_wind_gust)
	full_wind_dir = np.array(full_wind_dir)

	output_dict = {'BJD':full_bjd, 'BJD List':bjd_save, 'Flux':full_flux, 'Flux Error':full_flux_err, 
				'Exptime':full_exptime, 'Sky':full_sky, 'X':full_x, 'Y':full_y, 'Airmass':full_airmass,
				'FWHM X':full_fwhm_x, 'FWHM Y':full_fwhm_y, 'Humidity':full_humidity, 'Dome Humidity':full_dome_humidity, 'CCD Temp':full_ccd_temp, 'Dome Temp':full_dome_temp,
				'Focus':full_focus, 'Secondary Temp':full_sec_temp, 'Primary Temp':full_pri_temp,
				'Return Temp':full_ret_temp, 'Rod Temp':full_rod_temp, 'Cabinet Temp':full_cab_temp,
				'Instrument Temp':full_inst_temp, 'Temp': full_temp, 'Dewpoint':full_dewpoint, 
				'Sky Temp':full_sky_temp, 'Pressure':full_pressure, 'Return Pressure':full_ret_pressure,
				'Supply Pressure':full_supp_pressure, 'Hour Angle':full_ha, 'Dome Azimuth':full_dome_az,
				'Wind Speed':full_wind_spd, 'Wind Gust':full_wind_gust, 'Wind Direction':full_wind_dir}

	return  output_dict

def exposure_restrictor(data_dict, pos_hi, fwhm_hi, sky_hi, airmass_hi, flux_lo, dome_humidity_hi):

	position_mask = np.zeros(len(data_dict['BJD']), dtype='bool')
	ha_mask = np.zeros_like(position_mask)
	fwhm_mask = np.zeros_like(position_mask)
	airmass_mask = np.zeros_like(position_mask)
	sky_mask = np.zeros_like(position_mask)
	humidity_mask = np.zeros_like(position_mask) 
	flux_mask = np.zeros_like(position_mask)

	# dome_humidity_lo = limit_dict['dome_humidity'][0]
	# dome_humidity_hi = limit_dict['dome_humidity'][1]
	# airmass_lo = limit_dict['airmass'][0]
	# airmass_hi = limit_dict['airmass'][1]
	# fwhm_lo = limit_dict['fwhm'][0]
	# fwhm_hi = limit_dict['fwhm'][1]
	# sky_lo = limit_dict['sky'][0]
	# sky_hi = limit_dict['sky'][1]
	# pos_lo = limit_dict['pos'][0]
	# pos_hi = limit_dict['pos'][1]

	# start by masking exposures that do not depend on source measurements (i.e., flux, pixel position, and fwhm)

	# hour angle 
	use_inds = np.where(data_dict['Hour Angle'] > 0)[0]
	# exposure_mask[use_inds] = False
	ha_mask[use_inds] = True

	# airmass 
	use_inds = np.where((data_dict['Airmass'] > 1) & (data_dict['Airmass'] < airmass_hi))[0]	
	airmass_mask[use_inds] = True

	# sky 
	sky = np.median(data_dict['Sky'],axis=0)
	use_inds = np.where((sky > 0) & (sky  < sky_hi))[0]	
	sky_mask[use_inds] = True

	# dome humidity
	use_inds = np.where((data_dict['Dome Humidity'] > 0) & (data_dict['Dome Humidity'] < dome_humidity_hi))[0]
	humidity_mask[use_inds] = True

	exposure_mask = ha_mask & airmass_mask & sky_mask & humidity_mask
	
	# now add in masks on flux, positions, and fwhm, excluding the data that have already been masked

	#raw flux
	flux = data_dict['Flux']
	norm_flux = (flux.T / np.median(flux[:, exposure_mask], axis=1)).T
	norm_flux_med = np.median(norm_flux, axis=0)
	use_inds = np.where(norm_flux_med > flux_lo)[0]
	#exposure_mask[use_inds] = True
	flux_mask[use_inds] = True
	
	# positions
	x_excursions = abs(np.median((data_dict['X'].T - np.median(data_dict['X'][:,exposure_mask],axis=1)).T, axis=0))
	y_excursions = (np.median((data_dict['Y'].T - np.median(data_dict['Y'][:,exposure_mask],axis=1)).T, axis=0))
	use_inds = np.where((x_excursions>0)&
					 	(x_excursions<pos_hi)&
					    (y_excursions<0)& 
						(y_excursions<pos_hi))[0]
	position_mask[use_inds] = True

	# fwhm 
	fwhm_x = np.median(data_dict['FWHM X'],axis=0)
	use_inds = np.where((fwhm_x > 0) & (fwhm_x < fwhm_hi))[0]
	fwhm_mask[use_inds] = True

	# update the exposure mask to include the fwhm, position, and flux masks
	exposure_mask = position_mask & ha_mask & airmass_mask & fwhm_mask & sky_mask & humidity_mask & flux_mask

	output_dict = {}

	#start by updating the BJD list 
	bjd_list = []
	for i in range(len(data_dict['BJD List'])):
		use_inds = np.where((data_dict['BJD']>=data_dict['BJD List'][i][0])&(data_dict['BJD']<=data_dict['BJD List'][i][-1]))[0]
		exposure_mask_night = exposure_mask[use_inds]
		if sum(exposure_mask_night) > 1:
			bjd_list.append(data_dict['BJD'][use_inds][exposure_mask_night])
	flux = []
	flux_err = []
	sky = []
	x = []
	y = []
	fwhm_x = []
	fwhm_y = []
	for i in range(len(data_dict['Flux'])):
		flux.append(data_dict['Flux'][i][exposure_mask])
		flux_err.append(data_dict['Flux Error'][i][exposure_mask])
		sky.append(data_dict['Sky'][i][exposure_mask])
		x.append(data_dict['X'][i][exposure_mask])
		y.append(data_dict['Y'][i][exposure_mask])
		fwhm_x.append(data_dict['FWHM X'][i][exposure_mask])
		fwhm_y.append(data_dict['FWHM Y'][i][exposure_mask])
	flux = np.array(flux)
	flux_err = np.array(flux_err)
	sky = np.array(sky)
	x = np.array(x)
	y = np.array(y)
	fwhm_x = np.array(fwhm_x)
	fwhm_y = np.array(fwhm_y)
	
	output_dict['BJD'] = data_dict['BJD'][exposure_mask]
	output_dict['BJD List'] = bjd_list
	output_dict['Flux'] = flux
	output_dict['Flux Error'] = flux_err
	output_dict['Exptime'] = data_dict['Exptime'][exposure_mask]
	output_dict['Sky'] = sky
	output_dict['X'] = x
	output_dict['Y'] = y
	output_dict['Airmass'] = data_dict['Airmass'][exposure_mask]
	output_dict['FWHM X'] = fwhm_x
	output_dict['FWHM Y'] = fwhm_y
	output_dict['Humidity'] = data_dict['Humidity'][exposure_mask]
	output_dict['Dome Humidity'] = data_dict['Dome Humidity'][exposure_mask]
	output_dict['CCD Temp'] = data_dict['CCD Temp'][exposure_mask]
	output_dict['Dome Temp'] = data_dict['Dome Temp'][exposure_mask]
	output_dict['Focus'] = data_dict['Focus'][exposure_mask]
	output_dict['Secondary Temp'] = data_dict['Secondary Temp'][exposure_mask]
	output_dict['Primary Temp'] = data_dict['Primary Temp'][exposure_mask]
	output_dict['Return Temp'] = data_dict['Return Temp'][exposure_mask]
	output_dict['Rod Temp'] = data_dict['Rod Temp'][exposure_mask]
	output_dict['Cabinet Temp'] = data_dict['Cabinet Temp'][exposure_mask]
	output_dict['Instrument Temp'] = data_dict['Instrument Temp'][exposure_mask]
	output_dict['Temp'] = data_dict['Temp'][exposure_mask]
	output_dict['Dewpoint'] = data_dict['Dewpoint'][exposure_mask]
	output_dict['Sky Temp'] = data_dict['Sky Temp'][exposure_mask]
	output_dict['Pressure'] = data_dict['Pressure'][exposure_mask]
	output_dict['Return Pressure'] = data_dict['Return Pressure'][exposure_mask]
	output_dict['Supply Pressure'] = data_dict['Supply Pressure'][exposure_mask]
	output_dict['Hour Angle'] = data_dict['Hour Angle'][exposure_mask]
	output_dict['Dome Azimuth'] = data_dict['Dome Azimuth'][exposure_mask]
	output_dict['Wind Speed'] = data_dict['Wind Speed'][exposure_mask]
	output_dict['Wind Gust'] = data_dict['Wind Gust'][exposure_mask]
	output_dict['Wind Direction'] = data_dict['Wind Direction'][exposure_mask]
	
	return output_dict

def limit_generator(full_dict):
	'''
		Generates random limits on exposure parameters. 

		Inputs:
			full_dict (dict): Dictionary from make_global_lists.

		Returns:
			limit_dict (dict): Dictionary where each entry is a tuple specifiying the lower/upper boundary on the parameter in question.
	'''

	# set some boundaries based on the data actually contained in the full dictionary
	# it doesn't make sense to apply limits that fall outside of the actual range of measured parameter values, that would just waste time
	dome_humidity_lo = np.nanmin(full_dict['Dome Humidity'])
	dome_humidity_hi = np.nanmax(full_dict['Dome Humidity'])
	airmass_lo = np.nanmin(full_dict['Airmass'])
	airmass_hi = np.nanmax(full_dict['Airmass'])
	fwhm_lo = 1 # just set reasonable fwhm limits
	fwhm_hi = 5
	sky_lo = np.nanmin(np.nanmedian(full_dict['Sky'], axis=0))
	sky_hi = np.nanmax(np.nanmedian(full_dict['Sky'], axis=0))
	pos_lo = 0.5 # just set reasonable position limits
	pos_hi = 10
	
	
	limit_dict = {}
	limit_dict['dome_humidity'] = np.sort(np.random.uniform(low=dome_humidity_lo, high=dome_humidity_hi, size=2))
	limit_dict['airmass'] = np.sort(np.random.uniform(low=airmass_lo, high=airmass_hi, size=2))
	limit_dict['fwhm'] = np.sort(np.random.uniform(low=fwhm_lo, high=fwhm_hi, size=2))
	limit_dict['sky'] = np.sort(np.random.uniform(low=sky_lo, high=sky_hi, size=2)) 
	limit_dict['pos'] = np.sort(np.random.uniform(low=pos_lo, high=pos_hi, size=2))   
	
	limit_dict['dome_humidity'][0] = 0
	limit_dict['airmass'][0] = 0 
	limit_dict['fwhm'][0] = 0
	limit_dict['sky'][0] = 0 
	limit_dict['pos'][0] = 0
	return limit_dict

def log_likelihood(theta, x):
	pos_hi, fwhm_hi, sky_hi, airmass_hi, flux_lo, dome_humidity_hi = theta 
	restricted_dict = exposure_restrictor(x, pos_hi=pos_hi, fwhm_hi=fwhm_hi, sky_hi=sky_hi, airmass_hi=airmass_hi, flux_lo=flux_lo, dome_humidity_hi=dome_humidity_hi)
	
	if len(restricted_dict['BJD']) == 0:
		return -np.inf
	if len(restricted_dict['BJD List']) <= 1:
		return -np.inf

	z, y, yerr =  z_score_calculator(mearth_style_pat_weighted_flux(restricted_dict))
	if z.shape[1] == 0:
		return -np.inf

	return -0.5 * np.sum((y-1)**2/yerr**2)

def log_prior(theta):
	pos_hi, fwhm_hi, sky_hi, airmass_hi, flux_lo, dome_humidity_hi = theta 
	if (0 < pos_hi < 10) and (0 < fwhm_hi < 5) and (0 < sky_hi < 20) and (1 < airmass_hi < 1.6) and (0.8 < flux_lo < 1.0) and (0 < dome_humidity_hi < 100):
		return 0

	return -np.inf

def log_probability(theta, x):
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	
	prob = lp + log_likelihood(theta, x)
	if np.isnan(prob):
		return -np.inf 

	return lp + log_likelihood(theta, x)

if __name__ == '__main__':

	target_field = 'LP119-26'
	data_path = '/home/ptamburo/tierras/pat_scripts/SAME/output/'
	restore = True 
	bkg_type = '1d'
	ap_radius = 10
	#night_list = ['20240212', '20240213', '20240214', '20240215', '20240216']	
	night_list = ['20240212', '20240213', '20240214', '20240215','20240216', '20240217', '20240219', '20240229', '20240301', '20240302', '20240303', '20240304']	

	ap_radii = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

	# humidity_limit = 30
	# airmass_limit = 1.4
	# fwhm_limit = 2.5
	# sky_limit = 2.0
	position_limit = 2
	flux_limit = 0.95

	# identify sources in the field from Gaia
	stars = gaia_query(target_field, overwrite=False)

	# do photometry
	for i in range(len(night_list)):
		if os.path.exists(data_path+f'/photometry/{night_list[i]}'):
			print(f'Photometry output already exists for {night_list[i]}, skipping!')
		else:
			print(f'Doing {night_list[i]}')
			files = get_flattened_files(night_list[i], target_field, 'flat0000')
			photometry(target_field, files, stars, ap_radii, an_in=35, an_out=55, centroid=True, bkg_type=bkg_type)

	# either restore a previously defined cluster or use user-provided thresholds to define a new one
	if restore:
		ans = input('Enter cluster ID number (int): ')
		cluster_ind = int(ans) 
		same_stars = pd.read_csv(data_path+f'clusters/{cluster_ind}/SAME_cluster.csv')
		gaia_ids = np.array(same_stars['source_id'])
		cluster = []
		for i in range(len(gaia_ids)):
			cluster.append(np.where(stars['source_id'] == gaia_ids[i])[0][0])
		cluster = np.array(cluster)
		phot_path = data_path + 'photometry'
	else:
		existing_clusters = [int(i) for i in os.listdir(data_path+'clusters/')]
		if len(existing_clusters) == 0:
			cluster_ind = 0 
		else:
			cluster_ind = max(existing_clusters) + 1		

		# search for groups of stars that meet the following thresholds between at least one pair of sources:
		bp_rp_threshold = 1.5
		G_threshold = 3
		rp_threshold = 12
		distance_threshold = 2500
		max_rp = 17.75
		same_stars, cluster, all_stars = star_selection(target_field, stars, bp_rp_threshold=bp_rp_threshold, G_threshold=G_threshold, distance_threshold=distance_threshold, max_rp=max_rp)
		cluster = [i for i in cluster[0]]	

		# save a csv of the SAME stars
		if not os.path.exists(data_path+f'clusters/{cluster_ind}'):
			os.mkdir(data_path+f'clusters/{cluster_ind}')
		same_df = same_stars.to_pandas().to_csv(data_path+f'clusters/{cluster_ind}/SAME_cluster.csv')

		# save a .profile file containing the parameters that were used to create the cluster for record keeping
		with open(data_path+f'clusters/{cluster_ind}/cluster_{cluster_ind}.profile', 'w') as f:
			f.write(f'bp_rp_threshold = {bp_rp_threshold}\n')
			f.write(f'G_threshold = {G_threshold}\n')
			f.write(f'rp_threshold = {rp_threshold}\n')
			f.write(f'distance_threshold = {distance_threshold}\n')
			f.write(f'max_rp = {max_rp}')

		phot_path = data_path + 'photometry'

	# set up logger
	# anything at level DEBUG will be saved to the log file, while anything INFO or higher will be logged to the file and also printed to the console
	log_path = Path(f'/home/ptamburo/tierras/pat_scripts/SAME/output/clusters/{cluster_ind}/cluster{cluster_ind}_SAME.log')
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)	
	fh = logging.FileHandler(log_path, mode='w')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	logger.info(f'Running SAME on cluster {cluster_ind}')
	logger.info(f'Ap radius = {ap_radius}')

	# read in the photometry for this cluster
	global_list_dict = make_global_lists(cluster, phot_path, night_list=night_list, ap_radius=ap_radius, bkg_type=bkg_type)


	nwalkers = 32
	max_steps = 10000
	x = global_list_dict
	labels = ['pos_hi', 'fwhm_hi', 'sky_hi', 'airmass_hi', 'flux_lo', 'dome_humidity_hi']

	# initialize walkers	
	pos = []
	for i in range(len(labels)):
		if labels[i] == 'pos_hi':
			pos.append(np.random.uniform(low=0, high=10, size=nwalkers))
		if labels[i] == 'fwhm_hi':
			pos.append(np.random.uniform(low=0, high=5, size=nwalkers))
		if labels[i] == 'sky_hi':
			pos.append(np.random.uniform(low=0, high=20, size=nwalkers))
		if labels[i] == 'airmass_hi':
			pos.append(np.random.uniform(low=1, high=1.6, size=nwalkers))
		if labels[i] == 'flux_lo':
			pos.append(np.random.uniform(low=0.8, high=1.0, size=nwalkers))	
		if labels[i] == 'dome_humidity_hi':
			pos.append(np.random.uniform(low=0, high=100, size=nwalkers))	
	pos = np.array(pos).T
	nwalkers, ndim = pos.shape 

	# Set up the backend
	# Don't forget to clear it in case the file already exists
	filename = f'/home/ptamburo/tierras/pat_scripts/SAME/output/clusters/{cluster_ind}/cluster{cluster_ind}_mcmc.h5'
	backend = emcee.backends.HDFBackend(filename)
	backend.reset(nwalkers, ndim)
	
	# initialize the sampler
	# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=([x]))
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=([x]), backend=backend)
	# sampler.run_mcmc(pos, nsteps, progress=True)

	# track how avg autocorr time estimate changes 
	index = 0 
	autocorr = np.empty(max_steps)
	old_tau = np.inf

	corner_fig, corner_axes = plt.subplots(nrows=ndim, ncols=ndim, figsize=(10,10))
	chain_fig, chain_axes = plt.subplots(ndim, figsize=(10,7), sharex=True)

	for sample in sampler.sample(pos, iterations=max_steps, progress=True):
		if sampler.iteration % 100:
			continue
		# compute the autocorrelation time so far 
		# using tol=0 means we'll always get an estimate even if it isn't trustworthy 
		tau = sampler.get_autocorr_time(tol=0)
		autocorr[index] = np.mean(tau)
		index += 1

		# check convergence
		converged = np.all(tau * 100 < sampler.iteration)
		converged &= np.all(np.abs(old_tau -tau) /tau < 0.01)
		if converged:
			break
		old_tau = tau 

		# update plots 
		if index > 1:
			for a in chain_axes:
				a.cla()
			for a in corner_axes:
				for b in a:
					b.cla()

		samples = sampler.get_chain()
		flat_samples = sampler.get_chain(flat=True)
		for i in range(ndim):
			if ndim == 1:
				ax = chain_axes
			else:
				ax = chain_axes[i]
			ax.plot(samples[:,:,i], 'k', alpha=0.3)
			ax.set_ylabel(labels[i])
		ax.set_xlabel('step number')

		corner.corner(flat_samples, labels=labels, fig=corner_fig)
		
		plt.pause(0.1)
		

	samples = sampler.get_chain()
	# chain plot 
	fig, axes = plt.subplots(ndim, figsize=(10,7), sharex=True)
	for i in range(ndim):
		if ndim == 1:
			ax = axes
		else:
			ax = axes[i]
		ax.plot(samples[:,:,i], 'k', alpha=0.3)
		ax.set_ylabel(labels[i])
	ax[-1].set_xlabel('step number')

	# corner plot
	flat_samples = sampler.get_chain(flat=True)
	fig = corner.corner(flat_samples, labels=labels)
	breakpoint()

	


