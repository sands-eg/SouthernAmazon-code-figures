# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:38:51 2022

@author: s2261807

exploring CrIS isoprene data
"""

# =============================================================================
# import packages
# =============================================================================
import xarray as xr
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
# from sklearn.linear_model import LinearRegression, TheilSenRegressor
# import scipy.stats

# =============================================================================
# load data and create monthly xarray ds for 2012-2020
# =============================================================================
def open_isop(year):
    fn = f'R:/cris_isoprene/{year}_CrIS_monthly_Isoprene.nc'
    ds = xr.open_dataset(fn, decode_times=True)
    isop = ds['Isoprene'][:,::-1,:]
    ds.close()
    return isop

def open_pixels(year):
    fn = f'R:/cris_isoprene/{year}_CrIS_monthly_Isoprene.nc'
    ds = xr.open_dataset(fn, decode_times=True)
    px = ds['Num pixels'][:,::-1,:]
    ds.close()
    return px

isop_2013 = open_isop(2013)
lon = isop_2013.lon.values
lat = isop_2013.lat.values

isoprene = np.empty((108, 361, 576))
pixels = np.empty((108, 361, 576))

years = range(2012, 2021)

for y in years:
    i = y-2012
    isop = open_isop(y)
    isoprene[12*i:12*i+12,:,:] = isop.values
    
for y in years:
    i = y-2012
    px = open_pixels(y)
    pixels[12*i:12*i+12,:,:] = px.values

min_month = '2012-01'
max_month = '2020-12'
months = pd.period_range(min_month, max_month, freq = 'M').to_timestamp()

isoprene_ds = xr.Dataset(
    data_vars = dict(isoprene=(["time", "lat", "lon"], isoprene)),
    coords=dict(
        time =("time", months),
        lon=("lon", lon),
        lat=("lat", lat)))

pixels_ds = xr.Dataset(
    data_vars = dict(pixels=(["time", "lat", "lon"], pixels)),
    coords=dict(
        time =("time", months),
        lon=("lon", lon),
        lat=("lat", lat)))

# =============================================================================
# initial investigation - means, number of samples, spatial distribution
# =============================================================================

# global means/sums
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
axes.plot(isoprene_ds.time, isoprene_ds.isoprene.mean(axis=(1,2)), color = 'b', label = 'Isoprene (global mean)') #linestyle='None',
axes2 = axes.twinx()
axes2.plot(pixels_ds.time, pixels_ds.pixels.sum(axis=(1,2)), color = 'r', label = 'Number of pixels')
axes.legend(loc='upper left')
axes2.legend()

axes.set_ylabel('Mean global isoprene [molecules cm$^{-2}$]')
axes2.set_ylabel('Total number of pixels included')
axes.set_xlabel('year')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# global July means/sums
isop_mon = isoprene_ds['isoprene'][:,:,:].groupby('time.month').groups
july_idxs = isop_mon[7]
isop_july = isoprene_ds['isoprene'][:,:,:].isel(time=july_idxs)
pxls_july = pixels_ds['pixels'].isel(time=july_idxs)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
axes.plot(isop_july.time, isop_july.mean(axis=(1,2)), color = 'b', label = 'July isoprene') #linestyle='None',
axes2 = axes.twinx()
axes2.plot(pxls_july.time, pxls_july.sum(axis=(1,2)), color = 'r', label = 'Number of pixels')
axes.legend(loc='upper left')
axes2.legend(loc='upper right')
axes.set_ylabel('Mean global isoprene [mol cm$^{-2}$]')
axes2.set_ylabel('Total number of pixels included')
axes.set_xlabel('year')

isop_july_no2017 = np.delete(isop_july.values, 2017-2012, axis = 0)
july_mean_no2017 = np.nanmean(isop_july_no2017)
july_mean_2017 = isop_july[5,:,:].mean()
july_diff = july_mean_no2017 / july_mean_2017

july_2017_corrected = isop_july[5,:,:] * july_diff

july_corrected = isop_july.copy()
july_corrected[5,:,:] = july_2017_corrected

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
axes.plot(july_corrected.time, july_corrected.mean(axis=(1,2)), color = 'r', label = 'corrected') #linestyle='None',
axes.plot(isop_july.time, isop_july.mean(axis=(1,2)), color = 'b', label = 'original')
axes.legend(loc='upper left')
axes.set_ylabel('Mean global isoprene [mol cm$^{-2}$]')
axes.set_xlabel('year')

isoprene_ds['isoprene'][5,:,:] = july_2017_corrected
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# histograms
isop_mon = isoprene_ds['isoprene'][:,:,:].groupby('time.month').groups
july_idxs = isop_mon[7]
isop_july = isoprene_ds['isoprene'][:,:,:].isel(time=july_idxs)
pxls_july = pixels_ds['pixels'].isel(time=july_idxs)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8,6))
axes = axes.ravel()

data = isop_july[:,:,:]
dates = np.arange(2012, 2021)

for i,y in enumerate(dates):
    axes[i].set_title(y)
    axes[i].hist(data[i,:,:])

fig.tight_layout()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # negative values? - keep values in (random error at low isoprene column values?)
# negative = ma.masked_less(isoprene_ds['isoprene'], 0).mask

# isop_masked = isoprene_ds.where(isoprene_ds['isoprene'] > 0)

# negative2 = ma.masked_less(isop_masked['isoprene'], 0).mask

# # global July means/sums without negative values
# isop_mon = isop_masked['isoprene'][:,:,:].groupby('time.month').groups
# july_idxs = isop_mon[7]
# isop_july = isop_masked['isoprene'][:,:,:].isel(time=july_idxs)
# pxls_july = pixels_ds['pixels'].isel(time=july_idxs)

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
# axes.plot(isop_july.time, isop_july.mean(axis=(1,2)), color = 'b', label = 'July isoprene') #linestyle='None',
# axes2 = axes.twinx()
# axes2.plot(pxls_july.time, pxls_july.sum(axis=(1,2)), color = 'r', label = 'Number of pixels')
# axes.legend(loc='upper left')
# axes2.legend(loc='upper right')
# axes.set_ylabel('Mean global isoprene [mol cm$^{-2}$]')
# axes2.set_ylabel('Total number of pixels included')
# axes.set_xlabel('year')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# July comparison maps - contourf doesn't show correct colours???
projection = ccrs.PlateCarree() # ccrs.Robinson() # ccrs.PlateCarree() faster drawing?
transform = ccrs.PlateCarree()

longitude = isop_july.lon
latitude = isop_july.lat
data = np.zeros_like(isop_july)
climatology = isop_july.mean(axis=0)
for i in range(9):
    data[i,:,:] = isop_july[i,:,:] - climatology

# data = data.where(data < 0)
vmin = -1.4
vmax = 1.4
scaler = 1e15

cmap1 = plt.cm.get_cmap('RdBu_r')
levels = np.linspace(vmin*scaler, vmax*scaler, 8)

dates = np.arange(2012, 2021) #change_xr.time.values

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), subplot_kw={'projection':projection})
# axes.gridlines()
# im = axes.contourf(longitude, latitude, data[1,:,:], levels = levels, cmap = cmap1, extend = 'both', transform=transform)
# axes.coastlines()
# fig.suptitle('July 2013', fontsize = 16)
# # add colorbar
# cax = fig.add_axes([0.04, 0.02, 0.92, 0.02])
# fig.colorbar(im, cax=cax, orientation='horizontal', label = 'isoprene, difference from 2012-2020 July mean')

# fig.tight_layout()

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8), subplot_kw={'projection':projection})
axes = axes.ravel()

min_lon = -115    #60 #70 #-115 #-15 #-150 
max_lon = -30    #180 # 150 #-30 #65 #-50 
min_lat = -35    #30 #-15 #-35 #35 #25 
max_lat = 25    #90 #40 #25 #70 #65 

axes[0].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
# im = axes[0].imshow(data[0,:,:], cmap = cmap1, vmin = vmin*scaler, vmax = vmax*scaler, transform = transform)
im = axes[0].contourf(longitude, latitude, data[0,:,:], levels = levels, cmap = cmap1, extend = 'both', transform=transform)

for i,y in enumerate(dates):
    axes[i].set_title(y)
    axes[i].coastlines()
    axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    # axes[i].imshow(data[i,:,:], cmap = cmap1, vmin = vmin*scaler, vmax = vmax*scaler, transform = transform)
    axes[i].contourf(longitude, latitude, data[i,:,:], levels = levels, cmap = cmap1, extend = 'both', transform=transform)
fig.tight_layout()
fig.suptitle('July isoprene', fontsize = 16)
fig.subplots_adjust(top=0.92, right=0.9)

# add colorbar
cax = fig.add_axes([0.03, 0, 0.92, 0.02])
fig.colorbar(im, cax=cax, orientation='horizontal', label = 'isoprene, difference from 2012-2020 July mean')
# cax = fig.add_axes([0.91, 0.02, 0.02, 0.92])
# fig.colorbar(im, cax=cax, orientation='vertical', label = 'isoprene, difference from 2012-2020 July mean')
# # cax = fig.add_axes([0.1, 0, 0.8, 0.02])
# fig.colorbar(im, cax=cax, orientation='horizontal', label = 'x1.0E17 molecules/cm2')
fig.tight_layout()

# fig.savefig('M:\\figures\\atm_chem\\CO\\co_tc_annual_change_clim.png', dpi = 300)



# =============================================================================
# seasonal cycle
# =============================================================================
isop = isoprene_ds['isoprene'][:,:,:]
isop_monthly = isoprene_ds['isoprene'][:,:,:].groupby('time.month').mean()

### seaborn heatmap showing monthly mean isoprene for 2012-2020
data = isop[:,:,:].mean(axis = (1,2))
isop_reshape = np.zeros((9,12))
for k, v in data.groupby('time.year'):  # group the dateframe by year
    if k == 2012:
        isop_reshape[k-2012,1:] = v[1:]
        isop_reshape[k-2012,0] = np.nan
    else:
        isop_reshape[k-2012,:] = v[:]

fig, axes = plt.subplots(figsize=(8, 5))
axes.set_title('isop molecules $cm^{-2}$', fontsize = 12)
axes = sns.heatmap(isop_reshape, cbar_kws = {'label':'isop molecules $cm^{-2}$'})
axes.set_ylabel('Year', fontsize = 10)
axes.set_xlabel('Month', fontsize = 10)
axes.set_yticklabels(range(2012, 2021), fontsize = 8)
axes.set_xticklabels(range(1,13), fontsize = 8)
# fig.savefig('M:\\figures\\atm_chem\\isop\\isop_global_monthly_mean_heatmap_2005-2018.png', dpi = 300)




fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
axes.plot(isop_monthly.month, isop_monthly.mean(axis=(1,2)), color = 'b', label = 'isoprene') #linestyle='None',
# axes2 = axes.twinx()
# axes2.plot(pxls_july.time, pxls_july.sum(axis=(1,2)), color = 'r', label = 'Number of pixels')
# axes.legend(loc='upper left')
# axes2.legend(loc='upper right')
axes.set_ylabel('Mean global isoprene (2012-2020) [mol cm$^{-2}$]')
# axes2.set_ylabel('Total number of pixels included')
axes.set_xlabel('Month')

# =============================================================================
# difference from climatology
# =============================================================================

## annual mean weighted by month function
# based on https://ncar.github.io/esds/posts/2021/yearly-averages-xarray/
def weighted_temporal_mean(ds, var):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

#     # Subset our dataset for our variable
#     obs = ds[var]

    # Setup our masking for nan values
    cond = var.isnull() #obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (ds * wgts).resample(time="AS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

    # Return the weighted average
    return obs_sum / ones_out

isoprene_annual = weighted_temporal_mean(isop, isop)
isop_clim = isoprene_annual.mean(axis=0)

projection = ccrs.PlateCarree() # ccrs.Robinson() # ccrs.PlateCarree() faster drawing?
transform = ccrs.PlateCarree()

longitude = isoprene_annual.lon
latitude = isoprene_annual.lat
data = isoprene_annual[:,:,:] - isop_clim

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8,6), subplot_kw={'projection':projection})
axes = axes.ravel()

# min_lon = -115    #60 #70 #-115 #-15 #-150 
# max_lon = -30    #180 # 150 #-30 #65 #-50 
# min_lat = -35    #30 #-15 #-35 #35 #25 
# max_lat = 25    #90 #40 #25 #70 #65 

vmin = -1.4
vmax = 1.4
scaler = 1e15

cmap1 = plt.cm.get_cmap('RdBu_r')
levels = np.linspace(vmin*scaler, vmax*scaler, 8)

dates = np.arange(2012, 2021) #change_xr.time.values

#axes[0].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
im = axes[0].contourf(longitude, latitude, data[0,:,:], levels = levels, cmap = cmap1, extend = 'both', transform=transform)

for i,y in enumerate(dates):
    axes[i].set_title(y)
    axes[i].coastlines()
    #axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    axes[i].contourf(longitude, latitude, data[i,:,:], levels = levels, cmap = cmap1, extend = 'both', transform=transform)

fig.suptitle('Annual mean isoprene', fontsize = 16)
fig.tight_layout()
fig.subplots_adjust(top=0.92, right=0.9)

# add colorbar
cax = fig.add_axes([0.91, 0.02, 0.02, 0.92])
fig.colorbar(im, cax=cax, orientation='vertical', ) #label = 'x1.0E15 molecules/cm2'

# =============================================================================
# cropped data (lat lon)
# =============================================================================
def crop_data(var):
    ''' Function to crop data to area of interest:
        South America ()
        50-70 degrees W, 5-25 degrees S'''
    # mask_lon = (var.lon >= -83) & (var.lon <= -33)
    # mask_lat = (var.lat >= -56.5) & (var.lat <= 13.5)
    mask_lon = (var.lon >= -70) & (var.lon <= -50)
    mask_lat = (var.lat >= -25) & (var.lat <= -5)
    # mask_lon = (var.lon >= -180) & (var.lon <= 180)
    # mask_lat = (var.lat >= 0) & (var.lat <= 90)
    var_crop = var.where(mask_lon & mask_lat, drop=True)
    return var_crop

isop_crop = crop_data(isoprene_ds['isoprene'][:-2,:,:].groupby('time.month').mean())
isop_monthly = isop_crop
# isop_monthly = isop_crop['isoprene'][:,:,:].groupby('time.month').mean()

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
axes.plot(isop_monthly.month, isop_monthly.mean(axis=(1,2)), color = 'b', label = 'isoprene') #linestyle='None',
# axes2 = axes.twinx()
# axes2.plot(pxls_july.time, pxls_july.sum(axis=(1,2)), color = 'r', label = 'Number of pixels')
# axes.legend(loc='upper left')
# axes2.legend(loc='upper right')
axes.set_ylabel('Mean S. Amazon isoprene (2012-2018) [mol cm$^{-2}$]')
# axes2.set_ylabel('Total number of pixels included')
axes.set_xlabel('Month')

# =============================================================================
# removing outliers
# =============================================================================

# get mean values and standard deviations
monthly_means = isoprene_ds['isoprene'][:,:,:].groupby('time.month').mean()

monthly_SD =isoprene_ds['isoprene'][:,:,:].groupby('time.month').std()

# outlier_mask = np.ma.masked_outside()

def mask_outliers(data):
    '''

    Parameters
    ----------
    data : array
        Monthly isoprene data for one year.

    Returns
    -------
    data_masked

    '''
    under = ma.masked_less(data, monthly_means-2*monthly_SD).mask
    over = ma.masked_greater(data, monthly_means+2*monthly_SD).mask
    
    data_masked = data[under or over]
            
    return data_masked

masked_2017 = mask_outliers(isoprene[60:72])

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
axes.plot(monthly_means.month, monthly_means, color = 'blue')
axes.fill_between(monthly_means.month, monthly_means-2*monthly_SD, monthly_means+2*monthly_SD, color = 'grey') #linestyle='None',

# axes2 = axes.twinx()
# axes2.plot(pxls_july.time, pxls_july.sum(axis=(1,2)), color = 'r', label = 'Number of pixels')
# axes.legend(loc='upper left')
# axes2.legend(loc='upper right')
# axes.set_ylabel('Mean global isoprene [mol cm$^{-2}$]')
# axes2.set_ylabel('Total number of pixels included')
# axes.set_xlabel('year')


# remove values over 2 SD