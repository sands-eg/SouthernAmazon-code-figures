# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:10:52 2023

@author: s2261807

testing relationship derived from GFED4 and NO2 for 2005-2016
on GFED5 and NO2 for 2017-2020
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy.ma as ma
from sklearn.linear_model import TheilSenRegressor
import statsmodels.api as sm
from scipy.stats import spearmanr

# =============================================================================
# load data
# =============================================================================

# burned area (GFED 4; 2001-2016); error in metadata - unit is fraction of cell, not %
# fn = 'R:\\gfed\\monthly_1degree_sum_2001-2016.nc'
# ds = xr.open_dataset(fn, decode_times=True)
# fire = ds['Burned area']*100 
# fire['time'] = pd.date_range('2001-01-01', '2016-12-31', freq = 'MS')
# ds.close()
fn = 'R:\\gfed\\GFED4_BAkm2_2001-2016.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire = ds['Burned Area']
fire['time'] = pd.date_range('2001-01-01', '2016-12-31', freq = 'MS')
ds.close()

# burned area GFED5
fn = 'R:\gfed\GFED5\GFED5_totalBA_2001-2020.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire2 = ds['__xarray_dataarray_variable__']
fire2['time'] = pd.date_range('2001-01-01', '2020-12-31', freq = 'MS')

ds.close()
## DEM
fn = 'R:/DEM/SAM_DEM.nc'
ds = xr.open_dataset(fn)
dem = ds['SAM_DEM']
ds.close() 

### NO2
fn = 'R:/OMI_NO2/omi_no2_mm_2005_2020_masked.nc'
ds = xr.open_dataset(fn, decode_times=True)
no2 = ds['no2'][:,:,:]
ds.close()


# =============================================================================
# define functions
# =============================================================================
def spatial_weighted_average(data, weights):
    length = data.shape[0]
    weighted = np.ones(length)
    for i in range(length):
        weighted[i] = np.ma.average(np.ma.masked_invalid(data[i,:,:]), weights = weights)
    return weighted

def crop_data(var):
    ''' Function to crop data to area of interest:
        South America ()
        50-70 degrees W, 5-25 degrees S'''
   
    ### Southern Amazon
    mask_lon = (var.lon >= -70) & (var.lon <= -50)
    mask_lat = (var.lat >= -25) & (var.lat <= -5)

    var_crop = var.where(mask_lon & mask_lat, drop=True)
    return var_crop

def get_month_data(months, data):
    mon = data.groupby('time.month').groups
    mon_idxs = np.zeros(int(data.shape[0]/4))
    j = int(len(mon_idxs)/3)
    mon_idxs[:j] = mon[months[0]]
    mon_idxs[j:2*j] = mon[months[1]]
    mon_idxs[2*j:] = mon[months[2]]
    mon_idxs = np.sort(mon_idxs).astype(int)
    mon_data = data.isel(time=mon_idxs)
    return mon_data

def crop_ref_sector(var):
    ''' Function to crop data to area of interest:
        South America ()
        50-70 degrees W, 5-25 degrees S'''
    # mask_lon = (var.lon >= -83) & (var.lon <= -33)
    # mask_lat = (var.lat >= -56.5) & (var.lat <= 13.5)
    mask_lon = (var.lon >= -140) & (var.lon <= -100)
    mask_lat = (var.lat >= -30) & (var.lat <= 0)
    var_crop = var.where(mask_lon & mask_lat, drop=True)
    return var_crop

def weighted_temporal_mean(ds):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)
 
    # Setup our masking for nan values
    cond = ds.isnull() #obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (ds * wgts).resample(time="AS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

    # Return the weighted average
    return obs_sum / ones_out

def mask_data(data, yes_fire, no_fire):
    data_fires = xr.zeros_like(data)
    data_no_fires = xr.zeros_like(data)
    
    if len(data.shape) == 2:
        data_fires[:,:] = data[:,:].where(yes_fire)
        data_no_fires[:,:] = data[:,:].where(no_fire)
    elif len(data.shape) == 3:
        for i in range(len(data[:,0,0])):
            data_fires[i,:,:] = data[i,:,:].where(yes_fire[i,:,:])
            data_no_fires[i,:,:] = data[i,:,:].where(no_fire[i,:,:])
    elif len(data.shape) == 4:
        for i in range(len(data[:,0,0,0])):
            for j in range(len(data[0,:,0,0])):
                data_fires[i,j,:,:] = data[i,j,:,:].where(yes_fire[i,:,:])
                data_no_fires[i,j,:,:] = data[i,j,:,:].where(no_fire[i,:,:])
    else: print('Invalid shape of data')
    return data_fires, data_no_fires

def get_lai_dist(atm_comp, lc, perc):
    if perc == 15:
        cover = ma.masked_less_equal(lc, perc).mask.flatten()
    else:
        cover = ma.masked_inside(lc, perc - 4.9, perc).mask.flatten()
    
    atm_comp_1d = atm_comp.values.flatten()   
    # atm_comp_1d = atm_comp.flatten()   # use for hcho
    if np.any(cover) == False:
        cover = np.zeros_like(atm_comp_1d, dtype = np.bool)
    atm_comp_dist = atm_comp_1d[cover]
    return atm_comp_dist

def get_atm_dist(atm_comp, lc, perc):
    if perc == 5:
        cover = ma.masked_less_equal(lc, perc).mask.flatten()
    else:
        cover = ma.masked_inside(lc, perc - 4.9, perc).mask.flatten()
    
    atm_comp_1d = atm_comp.values.flatten()   
    # atm_comp_1d = atm_comp.flatten()   # use for hcho
    atm_comp_dist = atm_comp_1d[cover]
    return atm_comp_dist

def get_fire_dist(atm_comp, lc, perc):
    if perc == 5:
        cover = ma.masked_less_equal(lc, perc).mask.flatten()
    if perc ==50:
        cover = ma.masked_greater(lc, 45).mask.flatten()
    else:
        cover = ma.masked_inside(lc, perc - 4.9, perc).mask.flatten()
    
    atm_comp_1d = atm_comp.values.flatten()   
    # atm_comp_1d = atm_comp.flatten()   # use for hcho
    if np.any(cover) == False:
        cover = np.zeros_like(atm_comp_1d, dtype = np.bool)
    atm_comp_dist = atm_comp_1d[cover]
    return atm_comp_dist

def calc_statistic(data, stat = 'median'):
    data_means = np.zeros(10)
    if stat == 'mean':
        for i,a in enumerate(data):
            data_means[i]=a.mean()
    elif stat == 'median':
        for i,a in enumerate(data):
            data_means[i]=np.median(a)  
    else:
        print('invalid stat')
    
    return data_means

def mask_high_elev(data, high_elev):
    
    data_low = xr.zeros_like(data)
    
    if len(data.shape) == 2:
        data_low[:,:] = data[:,:].where(~high_elev)
    elif len(data.shape) == 3:
        for i in range(len(data[:,0,0])):
            data_low[i,:,:] = data[i,:,:].where(~high_elev)
    else: print('Invalid shape of data')
    return data_low

# =============================================================================
# weights for weighted spatial averages
# =============================================================================
#get weights
lon_edges = np.arange(-180, 180.1, 1)
lat_edges = np.arange(90, -90.1, -1)
# initialise variables
n_lons = len(lon_edges) - 1 # len() works for 1D arrays
n_lats = len(lat_edges) - 1 
R = 6371.0 # the radius of Earth in km
mdi = -999.99 # no data value
surface_area_earth = np.zeros((n_lats, n_lons)) + mdi
# get lat and lon in radians
lons_rad = lon_edges*(2*np.pi/360.0)
lats_rad = lat_edges*(2*np.pi/360.0)
# calculate surface area for lon lat grid 
# surface area = -1 * R^2 * (lon2 - lon1) * (sin(lat2) - sin(lat1)) # lat and lon in radians
for i in range(n_lons):
    for j in range(n_lats):
        term1 = R**2
        term2 = (lons_rad[i+1] - lons_rad[i])
        term3 = np.sin(lats_rad[j+1]) - np.sin(lats_rad[j])
        
        tmp_sa = -1*term1*term2*term3 # without -1 surface area comes out negative (lat in different order for MODIS?)
        surface_area_earth[j, i] = tmp_sa
spatial_weights = surface_area_earth / np.max(surface_area_earth)

spatial_weights = xr.DataArray(data = spatial_weights, coords = {"lat": fire.lat, "lon": fire.lon})

fire = fire.fillna(0)/surface_area_earth *100
fire2 = fire2.fillna(0)/surface_area_earth *100
# =============================================================================
# crop data to region of interest
# =============================================================================

fire_crop = crop_data(fire)
fire2_crop = crop_data(fire2)

dem_crop = crop_data(dem)

no2_crop = crop_data(no2)

weights_crop = crop_data(spatial_weights)
fire_spatial = spatial_weighted_average(fire_crop, weights_crop)
fire2_spatial = spatial_weighted_average(fire2_crop, weights_crop)

# =============================================================================
# separate data into wet and dry season
# =============================================================================
fire_wet = get_month_data([2,3,4], fire_crop)
fire_dry = get_month_data([8,9,10], fire_crop)
fire2_wet = get_month_data([2,3,4], fire2_crop)
fire2_dry = get_month_data([8,9,10], fire2_crop)

no2_wet = get_month_data([2,3,4], no2_crop)
no2_dry = get_month_data([8,9,10], no2_crop)


# =============================================================================
# create high-fire and low-fire data subsets = mask my burned area - what value should be chosen? - start with any fire in a given month
# =============================================================================
atmos_no = 4
atmos_dry = no2_dry 
atmos_wet = no2_wet 

labels = {1 : 'Isoprene', 2 : 'AOD' , 3 : 'Methanol', 4 : 'NO$_{2}$', 5 : 'CO', 6 : 'HCHO'}

units = {1 : '10$^{16}$ molecules cm$^{-2}$', 2 : 'at 0.47 $\mu$m', 3 : '[ppbv]', 4 : '[molecules cm$^{-2}$]',\
         5 : '[10$^{17}$ molecules cm$^{-2}$]', 6 : '[molecules cm$^{-2}$]'}


elev_boundary = 1000
high_elev = ma.masked_greater_equal(dem_crop, elev_boundary).mask

atmos_dry = mask_high_elev(atmos_dry, high_elev) 
atmos_wet = mask_high_elev(atmos_wet, high_elev)

   
# # ### keeping only data for months where both datasets are available
# min_time = xr.DataArray(data = [fire_dry.time.min().values, atmos_dry.time.min().values]).max().values
# min_year = min_time.astype(str).split('-')[0]
# max_time = xr.DataArray(data = [fire_dry.time.max().values, atmos_dry.time.max().values]).min().values
# max_year = max_time.astype(str).split('-')[0]
# fire_slice = fire_dry.sel(time=slice(min_year, max_year))
# fire_wet_slice = fire_wet.sel(time=slice(min_year, max_year))
    
## keeping only data for 2017-2020
min_time = xr.DataArray(data = [fire2_dry[-4*3:,:,:].time.min().values, atmos_dry.time.min().values]).max().values
min_year = min_time.astype(str).split('-')[0]
max_time = xr.DataArray(data = [fire2_dry[-4*3:,:,:].time.max().values, atmos_dry.time.max().values]).min().values
max_year = max_time.astype(str).split('-')[0]
fire_slice = fire2_dry[-4*3:,:,:].sel(time=slice(min_year, max_year))
fire_wet_slice = fire2_wet[-4*3:,:,:].sel(time=slice(min_year, max_year))

atmos_dry_slice = atmos_dry.sel(time=slice(min_year, max_year))
atmos_wet_slice = atmos_wet.sel(time=slice(min_year, max_year))


# =============================================================================
# Create dataframe - create 1 D lat and lon variables
# =============================================================================
atmos_wet_1d = atmos_wet_slice.values.ravel()
atmos_dry_1d = atmos_dry_slice.values.ravel()
# atmos_dry_1d = np.where(atmos_dry_1d < 10**17, atmos_dry_1d, np.nan) #for HCHO

fire_1d = fire_slice.values.ravel()
fire_wet_1d = fire_wet_slice.values.ravel()

# plt.scatter(lai_1d, atmos_dry_1d)
# plt.scatter(lc_1d, atmos_dry_1d)
# plt.scatter(fire_1d, atmos_dry_1d)

### create pandas dataframe to hold data of interest
# for_export = {'Atmos': atmos_dry_1d, 'Fire': fire_1d}
# data_export = pd.DataFrame(for_export)
# np.savetxt('C:/Users/s2261807/Documents/no2_burned_area.txt', data_export.values)

data = {'Atmos': atmos_dry_1d, 'Atmos_wet': atmos_wet_1d, 'Fire': fire_1d*2/3, 'Fire_wet': fire_wet_1d,\
        'Atmos_log': np.log(atmos_dry_1d),'Fire_log': np.log(fire_1d*2/3), 'Fire_wet_log': np.log(fire_wet_1d)} #, 'expFire': np.exp(fire_1d)}

data_pd = pd.DataFrame(data).replace(-np.Inf, np.nan).dropna()



print(spearmanr(data_pd['Atmos'], data_pd['Fire']))
print(spearmanr(data_pd['Atmos_log'], data_pd['Fire_log']))


# =============================================================================
# Figure comparing relationship from GFED4 to 2017-2020 data
# =============================================================================
Xi_all = data_pd['Fire_log']
# Xi = data_pd[['LC', 'Fire']]
y_all = data_pd['Atmos_log']

import scipy.stats
sorted_fire = data_pd.sort_values('Fire')

# bin_edges = np.arange(0, 500/1000, 25/1000) # multiplied by 10?
# bin_edges = np.append(bin_edges, 0.5)

bin_edges = np.arange(0, 0.5, 0.025) # multiplied by 10?
bin_edges = np.append(bin_edges, 0.5)

test_atm = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], bins= bin_edges)[0]
test_fire = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], bins= bin_edges)[0]
test_error = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], statistic = 'std', bins= bin_edges)[0] \
    / np.sqrt(scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], statistic = 'count', bins= bin_edges)[0])
fire_error = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], statistic = 'std', bins= bin_edges)[0] \
    / np.sqrt(scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], statistic = 'count', bins= bin_edges)[0])

err_max = test_atm + test_error
err_min = test_atm - test_error

ferr_max = test_fire + fire_error
ferr_min = test_fire - fire_error

Xi = np.log(test_fire)
# Xi = data_pd[['LC', 'Fire']]
y = np.log(test_atm)

log_min = np.log(err_min)
log_max = np.log(err_max)
log_errors  =np.zeros((2, len(log_min)))
log_errors[0,:] = y - log_min 
log_errors[1,:] = log_max - y

flog_min = np.log(ferr_min)
flog_max = np.log(ferr_max)
flog_errors  =np.zeros((2, len(flog_min)))
flog_errors[0,:] = Xi - flog_min 
flog_errors[1,:] = flog_max - Xi

par1 = 36.1
par2 = 0.261

resid_WLS = np.zeros(20)
for i in range(20):
    resid_WLS[i] = y[i] - (par1 + Xi[i]*par2)


cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12*cm, 18*cm))
axes = axes.ravel()
axes[0].scatter(Xi_all, y_all, s = 2, c = 'grey')#c = 'grey')
# axes[0].plot(Xi_all, y_pred_all, c ='navy')
axes[0].set_ylabel(f'Natural log of {labels[atmos_no]} \n{units[atmos_no]}', size = 10)
axes[0].set_xlabel('Natural log of burned area', size = 10)

axes[1].errorbar(Xi, y,  log_errors, linestyle = '',\
                 marker = 'o', markersize = 2, capsize = 4, c = 'grey')
axes[1].plot(Xi, par1 + Xi*par2, c ='navy')
axes[1].set_ylabel(f'Natural log of {labels[atmos_no]} \n{units[atmos_no]}', size = 10)
axes[1].set_xlabel('Natural log of burned area', size = 10)

axes[2].hist(resid_WLS, color = 'grey')
axes[2].set_ylabel('Residual count', size = 10)
axes[2].set_xlabel('log(NO2) residuals for the WLS regression', size = 10)


# axes.tick_params(axis='both', which='major', labelsize=14)
# fig.text(0.23, 0.89, f'y = {reg.coef_[1]:.2e}*x + {reg.coef_[0]:.2e}  \nR$^{2}$ = {score:.2f}', size = 12, c = 'navy')
# fig.text(0.23, 0.57, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = {reg_weighted.rsquared:.2f}', size = 12, c = 'navy')

# fig.text(0.23, 0.95, r'$log_{e}(NO_2) = 1.82 \times 10^{-1} \times log_{e}(BA) \plus 18$', size = 11, c = 'navy') 
# fig.text(0.23, 0.91, f'R$^{2}$ = {score:.2f}', size = 11, c = 'navy')
# fig.text(0.23, 0.62, r'$log_{e}(NO_2) = 2.59 \times 10^{-1} \times log_{e}(BA) \plus 36.7$', size = 11, c = 'navy') 
# fig.text(0.23, 0.58, f'R$^{2}$ = {reg_weighted.rsquared:.2f}', size = 11, c = 'navy')


fig.text(0.03, 0.95, '(a)', size = 14)
fig.text(0.03, 0.65, '(b)', size = 14)
fig.text(0.03, 0.33, '(c)', size = 14)

fig.tight_layout()
# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/GFED5_relationship_test.png', dpi = 300)


#### R2 calculation
unexplained = np.nansum(resid_WLS**2)
tot_var = np.nansum((y - np.nanmean(y))**2)
r2 = 1 - unexplained/tot_var

# =============================================================================
# binning - alternative method
# =============================================================================
Xi_all = data_pd['Fire_log']
# Xi = data_pd[['LC', 'Fire']]
y_all = data_pd['Atmos_log']

import scipy.stats
sorted_fire = data_pd.sort_values('Fire')

bin_edges = np.arange(0, 500/10000, 25/10000)
bin_edges = np.append(bin_edges, 0.5)

test_atm = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], bins= bin_edges)[0]
test_fire = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], bins= bin_edges)[0]
test_error = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], statistic = 'std', bins= bin_edges)[0] \
    / np.sqrt(scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], statistic = 'count', bins= bin_edges)[0])
fire_error = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], statistic = 'std', bins= bin_edges)[0] \
    / np.sqrt(scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], statistic = 'count', bins= bin_edges)[0])



err_max = test_atm + test_error
err_min = test_atm - test_error

ferr_max = test_fire + fire_error
ferr_min = test_fire - fire_error

Xi = np.log(test_fire)
# Xi = data_pd[['LC', 'Fire']]
y = np.log(test_atm)
X = sm.add_constant(Xi)

### weighted least squares

mod_weighted = sm.WLS(y, X, weights = 1/test_error, missing = 'drop')
reg_weighted = mod_weighted.fit()
print(reg_weighted.summary())
y_pred = reg_weighted.predict(X)

log_min = np.log(err_min)
log_max = np.log(err_max)
log_errors  =np.zeros((2, len(log_min)))
log_errors[0,:] = y - log_min 
log_errors[1,:] = log_max - y

flog_min = np.log(ferr_min)
flog_max = np.log(ferr_max)
flog_errors  =np.zeros((2, len(flog_min)))
flog_errors[0,:] = Xi - flog_min 
flog_errors[1,:] = flog_max - Xi

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
# axes.errorbar(Xi, y, log_errors, flog_errors, linestyle = '',\
#                  marker = 'o', markersize = 2, capsize = 4, c = 'grey')
# # axes.scatter(Xi, y, marker = 'o', facecolors = 'none', edgecolor = 'grey')
# axes.plot(Xi, y_pred, label = 'Predicted', c = 'navy')
# axes.set_ylabel(f'Nat log of mean {labels[atmos_no]} {units[atmos_no]}', size = 16)
# axes.set_xlabel('Nat log of mean burned area [% grid cell area]', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'
# fig.text(0.3, 0.7, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}', size = 12, c = 'navy') # \nR$^{2}$ = 0.98
# # axes.set_ylim(ymin = 34.5, ymax = 36)



resid_WLS = np.zeros(20)
for i in range(20):
    resid_WLS[i] = y[i] - (reg_weighted.params[0] + Xi[i]*reg_weighted.params[1])



cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12*cm, 18*cm))
axes = axes.ravel()
axes[0].scatter(Xi_all, y_all, s = 2, c=data_pd['LC'], cmap='YlGnBu')#c = 'grey')
# axes[0].plot(Xi_all, y_pred_all, c ='navy')
axes[0].set_ylabel(f'Natural log of {labels[atmos_no]} \n{units[atmos_no]}', size = 10)
axes[0].set_xlabel('Natural log of burned area', size = 10)

axes[1].errorbar(Xi, y,  log_errors, linestyle = '',\
                 marker = 'o', markersize = 2, capsize = 4, c = 'grey')
axes[1].plot(Xi, reg_weighted.params[0] + Xi*reg_weighted.params[1], c ='navy')
axes[1].set_ylabel(f'Natural log of {labels[atmos_no]} \n{units[atmos_no]}', size = 10)
axes[1].set_xlabel('Natural log of burned area', size = 10)

axes[2].hist(resid_WLS, color = 'grey')
axes[2].set_ylabel('Residual count', size = 10)
axes[2].set_xlabel('log(NO2) residuals for the WLS regression', size = 10)


# axes.tick_params(axis='both', which='major', labelsize=14)
# fig.text(0.23, 0.89, f'y = {reg.coef_[1]:.2e}*x + {reg.coef_[0]:.2e}  \nR$^{2}$ = {score:.2f}', size = 12, c = 'navy')
# fig.text(0.23, 0.57, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = {reg_weighted.rsquared:.2f}', size = 12, c = 'navy')

fig.text(0.23, 0.95, r'$log_{e}(NO_2) = 1.82 \times 10^{-1} \times log_{e}(BA) \plus 18$', size = 11, c = 'navy') 
fig.text(0.23, 0.91, f'R$^{2}$ = {score:.2f}', size = 11, c = 'navy')
fig.text(0.23, 0.62, r'$log_{e}(NO_2) = 2.59 \times 10^{-1} \times log_{e}(BA) \plus 36.7$', size = 11, c = 'navy') 
fig.text(0.23, 0.58, f'R$^{2}$ = {reg_weighted.rsquared:.2f}', size = 11, c = 'navy')


fig.text(0.03, 0.95, '(a)', size = 14)
fig.text(0.03, 0.65, '(b)', size = 14)
fig.text(0.03, 0.33, '(c)', size = 14)

fig.tight_layout()
