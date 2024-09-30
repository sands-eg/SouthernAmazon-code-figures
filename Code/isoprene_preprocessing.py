# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:41:53 2022

@author: s2261807

CrIS isoprene preprocessing: July 2017 correction, outlier removal, July to October means

"""
# =============================================================================
# import packages
# =============================================================================
import xarray as xr
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# =============================================================================
# load data and create monthly xarray ds for 2012-2020
# =============================================================================
def open_isop(year):
    if year == 2017:
        fn = f'R:/cris_isoprene/{year}_CrIS_monthly_Isoprene_Julyfix.nc'
    else:
        fn = f'R:/cris_isoprene/{year}_CrIS_monthly_Isoprene.nc'
    ds = xr.open_dataset(fn, decode_times=True)
    isop = ds['Isoprene'][:,::-1,:]
    ds.close()
    return isop

def open_pixels(year):
    if year == 2017:
        fn = f'R:/cris_isoprene/{year}_CrIS_monthly_Isoprene_Julyfix.nc'
    else:
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

# # =============================================================================
# # Temporary July 2017 fix - global burden ratio method
# # =============================================================================


# # global July means/sums
# isop_mon = isoprene_ds['isoprene'][:,:,:].groupby('time.month').groups
# july_idxs = isop_mon[7]
# isop_july = isoprene_ds['isoprene'][:,:,:].isel(time=july_idxs)
# pxls_july = pixels_ds['pixels'].isel(time=july_idxs)

# # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
# # axes.plot(isop_july.time, isop_july.mean(axis=(1,2)), color = 'b', label = 'July isoprene') #linestyle='None',
# # axes2 = axes.twinx()
# # axes2.plot(pxls_july.time, pxls_july.sum(axis=(1,2)), color = 'r', label = 'Number of pixels')
# # axes.legend(loc='upper left')
# # axes2.legend(loc='upper right')
# # axes.set_ylabel('Mean global isoprene [mol cm$^{-2}$]')
# # axes2.set_ylabel('Total number of pixels included')
# # axes.set_xlabel('year')


# ### weighted means (by area)

# #get weights
# lon_edges = np.arange(-180.1, 180.1, 0.625)
# lat_edges = np.arange(90.25, -90.26, -0.5)
# # initialise variables
# n_lons = len(lon_edges) - 1 # len() works for 1D arrays
# n_lats = len(lat_edges) - 1 
# R = 6371.0 # the radius of Earth in km
# mdi = -999.99 # no data value
# surface_area_earth = np.zeros((n_lats, n_lons)) + mdi
# # get lat and lon in radians
# lons_rad = lon_edges*(2*np.pi/360.0)
# lats_rad = lat_edges*(2*np.pi/360.0)
# # calculate surface area for lon lat grid 
# # surface area = -1 * R^2 * (lon2 - lon1) * (sin(lat2) - sin(lat1)) # lat and lon in radians
# for i in range(n_lons):
#     for j in range(n_lats):
#         term1 = R**2
#         term2 = (lons_rad[i+1] - lons_rad[i])
#         term3 = np.sin(lats_rad[j+1]) - np.sin(lats_rad[j])
        
#         tmp_sa = -1*term1*term2*term3 # without -1 surface area comes out negative (lat in different order for MODIS?)
#         surface_area_earth[j, i] = tmp_sa

# spatial_weights = surface_area_earth / np.max(surface_area_earth)

# def spatial_weighted_average(data, weights):
#     length = data.shape[0]
#     weighted = np.ones(length)
#     for i in range(length):
#         weighted[i] = np.ma.average(np.ma.masked_invalid(data[i,:,:]), weights = weights)
#     return weighted

# weighted_original = spatial_weighted_average(isop_july, spatial_weights)



# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
# axes.plot(isop_july.time, weighted_original, color = 'b', label = 'July isoprene') #linestyle='None',
# axes2 = axes.twinx()
# axes2.plot(pxls_july.time, pxls_july.sum(axis=(1,2)), color = 'r', label = 'Number of pixels')
# axes.legend(loc='upper left')
# axes2.legend(loc='upper right')
# axes.set_ylabel('Mean global isoprene [mol cm$^{-2}$]')
# axes2.set_ylabel('Total number of pixels included')
# axes.set_xlabel('year')


# # correct July 2017
# isop_july_no2017 = np.delete(isop_july.values, 2017-2012, axis = 0)
# july_means_no2017 = np.zeros(8)
# for i in range(len(july_means_no2017)):
#     july_means_no2017[i] = np.ma.average(np.ma.masked_invalid(isop_july_no2017[i,:,:]), weights = spatial_weights)

# july_mean_no2017 = np.nanmean(july_means_no2017)
# july_mean_2017 = np.ma.average(np.ma.masked_invalid(isop_july[2017-2012,:,:]), weights = spatial_weights)

# july_R = july_mean_no2017 / july_mean_2017

# july_2017_corrected = isop_july[5,:,:] * july_R

# july_corrected = isop_july.copy()
# july_corrected[5,:,:] = july_2017_corrected

# weighted_corrected = spatial_weighted_average(july_corrected, spatial_weights)

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
# axes.plot(july_corrected.time, weighted_corrected, color = 'r', label = 'corrected') #linestyle='None',
# # axes.plot(isop_july.time, weighted_original, color = 'b', label = 'original')
# axes.legend(loc='upper left')
# axes.set_ylabel('Mean global isoprene [mol cm$^{-2}$]')
# axes.set_xlabel('year')

# # isoprene_ds['isoprene'][(2017-2012)*12+6,:,:] = july_2017_corrected
# # 
# clim_july = isop_july_no2017.mean(axis=0)
# data = (july_2017_corrected - clim_july) / clim_july * 100

# projection = ccrs.PlateCarree() # ccrs.Robinson() # ccrs.PlateCarree() faster drawing?
# transform = ccrs.PlateCarree()

# longitude = isop_july.lon
# latitude = isop_july.lat
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,8), subplot_kw={'projection':projection})

# vmin = -100
# vmax = 100
# scaler = 1

# cmap1 = plt.cm.get_cmap('RdBu_r')
# levels = np.linspace(vmin*scaler, vmax*scaler, 11)


# im = axes.contourf(longitude, latitude, data, levels = levels, cmap = cmap1, extend = 'both', transform=transform)

# axes.set_title('July 2017, adjusted based on global average burden')
# axes.coastlines()
# axes.contourf(longitude, latitude, data, levels = levels, cmap = cmap1, extend = 'both', transform=transform)

# # add colorbar
# cax = fig.add_axes([0.04, 0.1, 0.92, 0.02])
# fig.colorbar(im, cax=cax, orientation='horizontal', label = 'isoprene, difference from 2012-2020 July mean as % of mean July value')

# fig.tight_layout()

# # =============================================================================
# # June to August interpolation? (=> June and August average)
# # =============================================================================

# # INPUTS
# isop_new = isoprene_ds.copy()

# isop_mon = isoprene_ds['isoprene'][:,:,:].groupby('time.month').groups
# july_idxs = isop_mon[7]
# isop_july = isoprene_ds['isoprene'][:,:,:].isel(time=july_idxs)

# june_idxs = isop_mon[6]
# isop_june = isoprene_ds['isoprene'][:,:,:].isel(time=june_idxs)

# aug_idxs = isop_mon[8]
# isop_aug = isoprene_ds['isoprene'][:,:,:].isel(time=aug_idxs)

# july2017 = isop_july[2017-2012,:,:]
# june2017 = isop_june[2017-2012,:,:]
# aug2017 = isop_aug[2017-2012,:,:]

# new_july2017 = (june2017 + aug2017)/2

# ### weighted means (by area)

# #get weights
# lon_edges = np.arange(-180.1, 180.1, 0.625)
# lat_edges = np.arange(90.25, -90.26, -0.5)
# # initialise variables
# n_lons = len(lon_edges) - 1 # len() works for 1D arrays
# n_lats = len(lat_edges) - 1 
# R = 6371.0 # the radius of Earth in km
# mdi = -999.99 # no data value
# surface_area_earth = np.zeros((n_lats, n_lons)) + mdi
# # get lat and lon in radians
# lons_rad = lon_edges*(2*np.pi/360.0)
# lats_rad = lat_edges*(2*np.pi/360.0)
# # calculate surface area for lon lat grid 
# # surface area = -1 * R^2 * (lon2 - lon1) * (sin(lat2) - sin(lat1)) # lat and lon in radians
# for i in range(n_lons):
#     for j in range(n_lats):
#         term1 = R**2
#         term2 = (lons_rad[i+1] - lons_rad[i])
#         term3 = np.sin(lats_rad[j+1]) - np.sin(lats_rad[j])
        
#         tmp_sa = -1*term1*term2*term3 # without -1 surface area comes out negative (lat in different order for MODIS?)
#         surface_area_earth[j, i] = tmp_sa

# spatial_weights = surface_area_earth / np.max(surface_area_earth)

# def spatial_weighted_average(data, weights):
#     length = data.shape[0]
#     weighted = np.ones(length)
#     for i in range(length):
#         weighted[i] = np.ma.average(np.ma.masked_invalid(data[i,:,:]), weights = weights)
#     return weighted

# weighted_new = np.ma.average(np.ma.masked_invalid(new_july2017), weights = spatial_weights)

# isop_july_no2017 = np.delete(isop_july.values, 2017-2012, axis = 0)
# july_means_no2017 = np.zeros(8)
# for i in range(len(july_means_no2017)):
#     july_means_no2017[i] = np.ma.average(np.ma.masked_invalid(isop_july_no2017[i,:,:]), weights = spatial_weights)
# july_mean_no2017 = np.nanmean(july_means_no2017)

# R = july_mean_no2017 / weighted_new

# print(aug2017.mean())      
# print(new_july2017.mean())


# isoprene_ds['isoprene'][(2017-2012)*12+6,:,:] = new_july2017
# =============================================================================
# removing outliers
# =============================================================================

# get mean values and standard deviations
monthly_means = isoprene_ds['isoprene'][:,:,:].groupby('time.month').mean()

monthly_SD =isoprene_ds['isoprene'][:,:,:].groupby('time.month').std()

def mask_outliers(data, month):
    '''

    Parameters
    ----------
    data : array
        Monthly isoprene data for one year.

    Returns
    -------
    data_masked

    '''
    under = ma.masked_greater(data, monthly_means[month-1,:,:]-2*monthly_SD[month-1,:,:]).mask
    over = ma.masked_less(data, monthly_means[month-1,:,:]+2*monthly_SD[month-1,:,:]).mask
    
    data_masked = np.where(under, data, np.nan)
    data_masked = np.where(over, data_masked, np.nan)
            
    return data_masked

masked_2012 = mask_outliers(isoprene[(2017-2012)*12+6,:,:], 7)

# plt.hist(isoprene[(2017-2012)*12+6,:,:])
# plt.hist(masked_2012)


isoprene_masked = np.zeros_like(isoprene_ds['isoprene'][:,:,:])
isop_mon = isoprene_ds['isoprene'][:,:,:].groupby('time.month').groups

for m in range(12):
    july_idxs = isop_mon[m+1]
    data = isoprene_ds['isoprene'][:,:,:].isel(time=july_idxs)
    for y in range(9):
        masked_data = mask_outliers(data[y], m+1)
        isoprene_masked[y*12+m, :, :] = masked_data
 
isoprene_masked = xr.DataArray(
    data=isoprene_masked,
    dims=["time", "lat", "lon"],
    coords=dict(
        lon=lon,
        lat=lat,
        time=isoprene_ds['isoprene'].time,
    ),
    attrs=dict(
        description="Isoprene with masked outliers (3 SD)",
        units="mol/cm^2",
    ),
)       

# histograms 
#outliers removed
isop_mon = isoprene_masked[:,:,:].groupby('time.month').groups
july_idxs = isop_mon[7]
isop_july = isoprene_masked[:,:,:].isel(time=july_idxs)
pxls_july = pixels_ds['pixels'].isel(time=july_idxs)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8,6))
axes = axes.ravel()

data = isop_july[:,:,:]
dates = np.arange(2012, 2021)

for i,y in enumerate(dates):
    axes[i].set_title(y)
    axes[i].hist(data[i,:,:])

fig.tight_layout()
 
#original
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


# =============================================================================
# save data with corrected July 2017 and outliers removed to nc file
# =============================================================================
data = isoprene_masked.rename('isoprene')
data.to_netcdf('R:/cris_isoprene/2012-2020_CrIS_monthly_Isoprene_preprocessed_vAug23_2SDmask.nc')

# data = isoprene_ds['isoprene']
# data.to_netcdf('R:/cris_isoprene/2012-2020_CrIS_monthly_Isoprene_preprocessed_vAug23.nc')

# =============================================================================
# get summer (annual?) mean
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

isoprene_annual = weighted_temporal_mean(isoprene_masked, isoprene_masked)

data = isoprene_annual.rename('isoprene')
data.to_netcdf('R:/cris_isoprene/2012-2020_CrIS_annual_Isoprene_preprocessed_july20017interpolated.nc')
