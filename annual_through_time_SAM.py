# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:24:33 2023

Land cover change through time

@author: s2261807
"""

# =============================================================================
# import functions and load necessary data
# =============================================================================

### functions
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
# from datetime import datetime
from sklearn.linear_model import TheilSenRegressor

### data

## land cover, burned area

# # ### LC option A
# # broadleaf forest
# fn = 'R:\modis_lc\mcd12c1_broadleaf_1deg.nc'
# ds = xr.open_dataset(fn, decode_times=False)
# broadleaf = ds['broadleaf'][:]
# broadleaf['time'] = pd.date_range('2001', '2020', freq = 'Y')
# ds.close()

# # grasses and crops
# fn = 'R:\modis_lc\mcd12c1_grass_crop_1deg.nc'
# ds = xr.open_dataset(fn, decode_times=False)
# grass = ds['grass_crop'][:]
# grass['time'] = pd.date_range('2001', '2020', freq = 'Y')
# ds.close()

# # savanna
# fn = 'R:\modis_lc\mcd12c1_1deg_igbp_savanna_majority.nc'
# ds = xr.open_dataset(fn, decode_times=False)
# savanna = ds['Savanna'][:]/100
# savanna['time'] = pd.date_range('2001', '2020', freq = 'Y')
# ds.close()


## LC option B
fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=True)
broadleaf = (ds['Land_Cover_Type_1_Percent'][:,:,:,2] + ds['Land_Cover_Type_1_Percent'][:,:,:,4] )/100
broadleaf['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()

fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=False)
savanna = ds['Land_Cover_Type_1_Percent'][:,:,:,8:10].sum(axis=3)/100
savanna['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()

fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=False)
grass_mosaic = (ds['Land_Cover_Type_1_Percent'][:,:,:,10] + ds['Land_Cover_Type_1_Percent'][:,:,:,12] + ds['Land_Cover_Type_1_Percent'][:,:,:,14])/100
grass_mosaic['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()
grass = grass_mosaic

# burned area GFED5
fn = 'R:\gfed\GFED5\GFED5_totalBA_2001-2020.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire2 = ds['__xarray_dataarray_variable__']#*100 
fire2['time'] = pd.date_range('2001-01-01', '2020-12-31', freq = 'MS')
ds.close()

# isoprene
fn = 'R:/cris_isoprene/2012-2020_CrIS_monthly_Isoprene_1degree_vAug23_2SDmask.nc'
ds = xr.open_dataset(fn, decode_times=True)
isop = ds['isoprene'][:,:,:]
ds.close()

# methanol
fn = 'R:/methanol/methanol_1degree_2008-2018.nc'
ds = xr.open_dataset(fn, decode_times=True)
methanol = ds['methanol'][:,:,:]
ds.close()

stdev = np.std(methanol)
methanol = methanol.where(methanol < np.nanmean(methanol)+3*stdev, np.nan)

# HCHO
fn = 'R:/OMI_HCHO/OMI_HCHO_RSCVC_monthly_no_corrections.nc'
ds = xr.open_dataset(fn, decode_times=True)
hcho = ds['hcho_rs'][:-12,::-1,:]
ds.close()

# CO
fn = 'R:/mopitt_co/mopitt_co_totalcolumn_2001-2019_monthly.nc'
ds = xr.open_dataset(fn, decode_times=True)
co = ds['__xarray_dataarray_variable__']
ds.close()

# NO2
fn = 'R:/OMI_NO2/omi_no2_mm_2005_2020_masked.nc'
ds = xr.open_dataset(fn, decode_times=True)
# ds['time'] = pd.date_range('2005-01-01', '2020-12-31', freq = 'MS')
no2 = ds['no2'][:,:,:]
ds.close()

# AOD
fn = 'R:\modis_aod\mod08_aod_masked_2001-2019.nc'
ds = xr.open_dataset(fn, decode_times=True)
aod = ds['aod'][:,0,:,:]
# # mask_value = aod[0,0,0,0]
ds.close()

# burned area (GFED 4; 2001-2016); error in metadata - unit is fraction of cell, not %
fn = 'R:\\gfed\\monthly_1degree_sum_2001-2016.nc'
ds = xr.open_dataset(fn, decode_times=True)
ds.transpose('time', 'lat', 'lon')
fire = ds['Burned area']
fire['time'] = pd.date_range('2001-01-01', '2016-12-31', freq = 'MS')
ds.close()

## DEM
fn = 'R:/DEM/SAM_DEM.nc'
ds = xr.open_dataset(fn)
dem = ds['SAM_DEM']
ds.close()

## LAI
fn = 'R:\modis_lai\modis_lai_2001-2019.nc'
ds = xr.open_dataset(fn, decode_times=True)
lai = ds['LAI'][:]/10
ds.close()


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
surface_area = xr.DataArray(data = surface_area_earth, coords = {"lat": fire.lat, "lon": fire.lon})

fire = fire * surface_area
# =============================================================================
# Define additional functions
# =============================================================================

def spatial_weighted_average(data, weights):
    length = data.shape[0]
    weighted = np.ones(length)
    std = np.ones(length)
    errors = np.ones(length)
    for i in range(length):
        weighted[i] = np.ma.average(np.ma.masked_invalid(data[i,:,:]), weights = weights)
        std[i] = np.average((np.ma.masked_invalid(data[i,:,:]) - weighted[i])**2, weights=weights)
        errors[i] = std[i] / np.sqrt(np.count_nonzero(~np.isnan(data[i,:,:])))
    return weighted, errors

def crop_data(var):
    ''' Function to crop data to area of interest:
        South America ()
        50-70 degrees W, 5-25 degrees S'''
   
    ### Southern Amazon
    mask_lon = (var.lon >= -70) & (var.lon <= -50)
    mask_lat = (var.lat >= -25) & (var.lat <= -5)

    var_crop = var.where(mask_lon & mask_lat, drop=True)
    return var_crop

def mask_high_elev(data, high_elev):
    data_low = xr.zeros_like(data)
    
    if len(data.shape) == 2:
        data_low[:,:] = data[:,:].where(~high_elev)
    elif len(data.shape) == 3:
        for i in range(len(data[:,0,0])):
            data_low[i,:,:] = data[i,:,:].where(~high_elev)
    else: print('Invalid shape of data')
    return data_low

def get_atm_dist(atm_comp, lc, perc):
    if perc == 10:
        cover = ma.masked_less_equal(lc, perc).mask.flatten()
    else:
        cover = ma.masked_inside(lc, perc - 9.9, perc).mask.flatten()
    
    atm_comp_1d = atm_comp.values.flatten()   
    # atm_comp_1d = atm_comp.flatten()   # use for hcho
    if np.any(cover) == False:
        cover = np.zeros_like(atm_comp_1d, dtype = np.bool)
    atm_comp_dist = atm_comp_1d[cover]
    return atm_comp_dist

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

# =============================================================================
# Crop data
# =============================================================================
fire_crop = crop_data(fire)

broadleaf_crop = crop_data(broadleaf)
grass_crop = crop_data(grass)
savanna_crop = crop_data(savanna)
lai_crop = crop_data(lai)

dem_crop = crop_data(dem)
weights_crop = crop_data(spatial_weights)

fire2_crop = crop_data(fire2)


hcho_crop = crop_data(hcho)
co_crop = crop_data(co)
no2_crop = crop_data(no2)
aod_crop = crop_data(aod)
isop_crop = crop_data(isop)
methanol_crop = crop_data(methanol)

hcho_orig = hcho_crop
# =============================================================================
# Detrend HCHO
# =============================================================================
def crop_ref_sector(var):
    ''' Function to crop data to area of interest:
        Pacific Reference Sector'''
    mask_lon = (var.lon >= -140) & (var.lon <= -100)
    mask_lat = (var.lat >= -30) & (var.lat <= 0)
    var_crop = var.where(mask_lon & mask_lat, drop=True)
    return var_crop

pacific = crop_ref_sector(hcho)
pac_weights = crop_ref_sector(spatial_weights)

pacific_mean, pac_errors = spatial_weighted_average(pacific, pac_weights)


X = [i for i in range(0, len(pacific_mean))]
X = np.reshape(X, (len(X), 1))
y1 = pacific_mean
# Theil-Sen
model = TheilSenRegressor()
reg = model.fit(X, y1)
trend_TS = model.predict(X)
R2_ts = reg.score(X, y1)

trend_diff = trend_TS - trend_TS[0]


hcho_orig_mean, orig_error = spatial_weighted_average(hcho_orig, weights_crop)


detrended_hcho = np.zeros_like(hcho_orig)
for i in range(len(hcho_crop)):
    detrended_hcho[i,:,:] = hcho_orig[i,:,:] - trend_diff[i]
        
hcho_crop = xr.DataArray(data = detrended_hcho, coords = {"time": hcho_crop.time, "lat": hcho_crop.lat, "lon": hcho_crop.lon})

hcho_detrended_mean, detrended_error = spatial_weighted_average(hcho_crop,  weights_crop)

cm = 1/2.54
fig, ax = plt.subplots(figsize=(12*cm,9*cm))
  
ax.plot(hcho_crop.time, pacific_mean, lw = 1.5, c = 'grey', label = 'Background')
ax.plot(hcho_crop.time, trend_TS, ls = ':', c = 'black', label = 'Background trend')
ax.plot(hcho_crop.time, hcho_orig_mean, ls = '-', lw = 1.5, c = 'blue', label = 'SAM original')
ax.plot(hcho_crop.time, hcho_detrended_mean, lw = 1.5, c = 'maroon', label = 'SAM detrended')

ax.tick_params(axis='both', which='major', labelsize=8)
    
ax.set_ylabel('HCHO molecules cm$^{-2}$', fontsize = 8)
ax.set_xlabel('Time', fontsize = 8)
ax.legend(fontsize = 8, ncol = 2)#, loc = 'center right', bbox_to_anchor=(0.4, 0.3, 0.5, 0.5))


# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/hcho_detrending.pdf')

# =============================================================================
# mask high altitude areas
# =============================================================================
elev_boundary = 1000
high_elev = ma.masked_greater_equal(dem_crop, elev_boundary).mask

broadleaf_mask = mask_high_elev(broadleaf_crop, high_elev) 
grass_mask = mask_high_elev(grass_crop, high_elev) 
savanna_mask = mask_high_elev(savanna_crop, high_elev) 
fire_mask = mask_high_elev(fire_crop, high_elev)
sg_mask = savanna_mask + grass_mask

lai_mask = mask_high_elev(lai_crop, high_elev)

fire2_elev = mask_high_elev(fire2_crop, high_elev)

hcho_elev = mask_high_elev(hcho_crop, high_elev)
co_elev = mask_high_elev(co_crop, high_elev)
no2_elev = mask_high_elev(no2_crop, high_elev)
aod_elev = mask_high_elev(aod_crop, high_elev)
isop_elev = mask_high_elev(isop_crop, high_elev)
methanol_elev = mask_high_elev(methanol_crop, high_elev)



# =============================================================================
# use of pandas for annual mean calculations and error estimates?
# =============================================================================


def mean_error_lc(lc):
    bfc_1d = lc.values.ravel()
    time_bfc = lc.time.dt.year.values
    
    reshape_weights = np.zeros_like(lc)
    for a in range(lc.shape[0]):
        reshape_weights[a,:,:] = weights_crop

    weights1_1d = reshape_weights.ravel()
    
    reshape_time = np.zeros_like(lc)
    for a in range(lc.shape[0]):
        reshape_time[a,:,:] = time_bfc[a]
    
    time_1d = reshape_time.ravel()
    
    data = {'var': bfc_1d, 'time': time_1d, 'w1': weights1_1d}
    pd_bfc = pd.DataFrame(data).dropna()
    
    broadleaf_regional = np.zeros_like(time_bfc, dtype = 'float')
    broadleaf_error = np.zeros_like(time_bfc, dtype = 'float')
    for i, y in enumerate(time_bfc):
        annual_pd = pd_bfc[pd_bfc['time'] == y]
        w_mean = (annual_pd['var']*annual_pd['w1']).sum() / annual_pd['w1'].sum()
        std = np.sqrt(np.average((annual_pd['var'] - w_mean)**2, weights=annual_pd['w1'])) # annual_pd['var'] - w_mean
        count = len(annual_pd['var'])
        broadleaf_regional[i] = w_mean
        broadleaf_error[i] = std / np.sqrt(count)
    
    return broadleaf_regional, broadleaf_error

broadleaf_regional, broadleaf_error = mean_error_lc(broadleaf_mask)
sg_regional, sg_error = mean_error_lc(sg_mask)
lai_regional, lai_error = mean_error_lc(lai_mask)

def mean_error(atmos):
    atmos_1d = atmos.values.ravel()
    time_atmos = np.unique(atmos.time.dt.year.values)

    reshape_weights = np.zeros_like(atmos)
    for a in range(atmos.shape[0]):
        reshape_weights[a,:,:] = weights_crop
    w1_1d = reshape_weights.ravel()
    
    month_length = atmos.time.dt.days_in_month
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    
    reshape_weights = np.zeros_like(atmos)
    for a in range(atmos.shape[0]):
        reshape_weights[a,:,:] = wgts[a]
    w2_1d = reshape_weights.ravel()
    
    reshape_time = np.zeros_like(atmos)
    for a in range(atmos.shape[0]):
        reshape_time[a,:,:] = atmos.time.dt.year.values[a]
    time_1d = reshape_time.ravel()
    
    data = {'var': atmos_1d, 'time': time_1d, 'w1': w1_1d, 'w2':w2_1d}
    pd_data = pd.DataFrame(data).dropna()
    
    atmos_regional = np.zeros_like(time_atmos, dtype = 'float')
    atmos_error = np.zeros_like(time_atmos, dtype = 'float')
    for i, y in enumerate(time_atmos):
        annual_pd = pd_data[pd_data['time'] == y]
        w_mean = (annual_pd['var']*(annual_pd['w1']+annual_pd['w2'])).sum() / (annual_pd['w1'].sum() + annual_pd['w2'].sum())
        std = np.sqrt(np.average((annual_pd['var'] - w_mean)**2, weights=annual_pd['w1']+annual_pd['w2'])) #pd_data.std()['var'] # annual_pd['var'] - w_mean
        count = len(annual_pd['var'])
        atmos_regional[i] = w_mean
        atmos_error[i] = std / np.sqrt(count)
        # atmos_error[i] = std
        
    return atmos_regional, atmos_error


hcho_spatial, hcho_error = mean_error(hcho_elev)
co_spatial, co_error = mean_error(co_elev)
aod_spatial, aod_error = mean_error(aod_elev)
isop_spatial, isop_error = mean_error(isop_elev)
methanol_spatial, methanol_error = mean_error(methanol_elev)
no2_spatial, no2_error = mean_error(no2_elev)

fire_spatial, fire_error = mean_error(fire_mask)

# # =============================================================================
# # Get annual fire sum
# # =============================================================================
# fire_sum = fire_mask.groupby("time.year").sum()
# fire2_sum = fire2_elev.groupby("time.year").sum()

# # =============================================================================
# # annual means
# # =============================================================================

# isop_wet = get_month_data([2,3,4], isop_elev)
# isop_dry = get_month_data([8,9,10], isop_elev)
# isop_dry_mean = weighted_temporal_mean(isop_dry)
# isop_wet_mean = weighted_temporal_mean(isop_wet)

# co_wet = get_month_data([2,3,4], co_elev)
# co_dry = get_month_data([8,9,10], co_elev)
# co_dry_mean = weighted_temporal_mean(co_dry)
# co_wet_mean = weighted_temporal_mean(co_wet)

# no2_wet = get_month_data([2,3,4], no2_elev)
# no2_dry = get_month_data([8,9,10], no2_elev)
# no2_dry_mean = weighted_temporal_mean(no2_dry)
# no2_wet_mean = weighted_temporal_mean(no2_wet)

# aod_wet = get_month_data([2,3,4], aod_elev)
# aod_dry = get_month_data([8,9,10], aod_elev)
# aod_dry_mean = weighted_temporal_mean(aod_dry)
# aod_wet_mean = weighted_temporal_mean(aod_wet)

# methanol_wet = get_month_data([2,3,4], methanol_elev)
# methanol_dry = get_month_data([8,9,10], methanol_elev)
# methanol_dry_mean = weighted_temporal_mean(methanol_dry)
# methanol_wet_mean = weighted_temporal_mean(methanol_wet)


# hcho_wet = get_month_data([2,3,4], hcho_elev)
# hcho_dry = get_month_data([8,9,10], hcho_elev)
# hcho_dry_mean = weighted_temporal_mean(hcho_dry)
# hcho_wet_mean = weighted_temporal_mean(hcho_wet)



# hcho_annual = weighted_temporal_mean(hcho_elev)
# co_annual = weighted_temporal_mean(co_elev)
# aod_annual = weighted_temporal_mean(aod_elev)
# isop_annual = weighted_temporal_mean(isop_elev)
# no2_annual = weighted_temporal_mean(no2_elev)
# methanol_annual = weighted_temporal_mean(methanol_elev)
# fire_annual = weighted_temporal_mean(fire_mask)

# # =============================================================================
# # Regional means (sum in case of fires)
# # =============================================================================
# broadleaf_regional, broadleaf_error = spatial_weighted_average(broadleaf_mask, weights_crop)
# grass_regional, grass_error = spatial_weighted_average(grass_mask, weights_crop)
# savanna_regional, savanna_error = spatial_weighted_average(savanna_mask, weights_crop)
# lai_regional, lai_error = spatial_weighted_average(lai_mask, weights_crop)
# fire_regional, fire_error = spatial_weighted_average(fire_annual, weights_crop)
# # fire_regional = fire_sum.sum(axis=(1,2))
# # fire2_regional = fire2_sum.sum(axis=(1,2))


# hcho_spatial, hcho_error = spatial_weighted_average(hcho_annual, weights_crop)
# co_spatial, co_error = spatial_weighted_average(co_annual, weights_crop)
# aod_spatial, aod_error = spatial_weighted_average(aod_annual, weights_crop)
# isop_spatial, isop_error = spatial_weighted_average(isop_annual, weights_crop)
# methanol_spatial, methanol_error = spatial_weighted_average(methanol_annual, weights_crop)
# no2_spatial, no2_error = spatial_weighted_average(no2_annual, weights_crop)

# # hcho_wet_spatial = spatial_weighted_average(hcho_wet_mean, weights_crop)
# # co_wet_spatial = spatial_weighted_average(co_wet_mean, weights_crop)
# # aod_wet_spatial = spatial_weighted_average(aod_wet_mean, weights_crop)
# # isop_wet_spatial = spatial_weighted_average(isop_wet_mean, weights_crop)
# # methanol_wet_spatial = spatial_weighted_average(methanol_wet_mean, weights_crop)
# # no2_wet_spatial = spatial_weighted_average(no2_wet_mean, weights_crop)

# # hcho_dry_spatial = spatial_weighted_average(hcho_dry_mean, weights_crop)
# # co_dry_spatial = spatial_weighted_average(co_dry_mean, weights_crop)
# # aod_dry_spatial = spatial_weighted_average(aod_dry_mean, weights_crop)
# # isop_dry_spatial = spatial_weighted_average(isop_dry_mean, weights_crop)
# # methanol_dry_spatial = spatial_weighted_average(methanol_dry_mean, weights_crop)
# # no2_dry_spatial = spatial_weighted_average(no2_dry_mean, weights_crop)

# sg_regional = savanna_regional + grass_regional
# sg_error = np.sqrt(savanna_error**2 + grass_error**2)
# =============================================================================
# Plot
# =============================================================================
# year = np.arange(2001, 2020)

# fig, ax = plt.subplots(figsize=(13,9))

# ax.plot(year, broadleaf_regional*100, color = 'blue', label = 'Broadleaf')
# ax2 = ax.twinx()
# ax2.plot(year, grass_regional*100, color = 'orange', label = 'Grasslands')
# ax2.plot(year, savanna_regional*100, color = 'maroon', label = 'Savanna')
# ax2.plot(year, lai_regional*10, color = 'green', label = 'LAI x 10')

# year_ticks = np.arange(2001, 2020,2)
# ax.set_xticks(year_ticks)
# ax.set_ylabel('Broadleaf [%]', fontsize = 16)
# ax2.set_ylabel('Grassland or Savanna [%]', fontsize = 16)
# ax.set_xlabel('Year', fontsize = 16)
# fig.legend(fontsize = 14, loc = 'center right', bbox_to_anchor=(0.4, 0.3, 0.5, 0.5))


# # save figure
# # fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/MonteCarlo/{labels[atmos_no]}_{lc_label}_lineandbootstrap_noneabove{elev_boundary}.png', dpi = 300)


# year = np.arange(2001, 2017)
# year2 = np.arange(2001, 2021)
# fig, ax = plt.subplots(figsize=(13,9))

# ax.plot(year, fire_regional*10, color = 'maroon', label = 'GFED4 x10')
# # ax.plot(year2, fire2_regional, color = 'cyan', label = 'GFED5')

# year_ticks = np.arange(2001, 2020, 2)
# ax.set_xticks(year_ticks)
# ax.set_ylabel('Burned Area (km^2)', fontsize = 16)
# ax.set_xlabel('Year', fontsize = 16)
# ax.legend(fontsize = 14)


## atm comp
year_ticks = np.arange(2001, 2020, 5)
data = [broadleaf_regional*100, sg_regional*100, lai_regional, fire_spatial, isop_spatial/10**15, \
        methanol_spatial, hcho_spatial/10**16, co_spatial, no2_spatial/10**15, aod_spatial] 
errors = [broadleaf_error*100, sg_error*100, lai_error, fire_error, isop_error/10**15, methanol_error, hcho_error/10**16, co_error, no2_error/10**15, aod_error] 

# data_dry = [broadleaf_regional*100, lai_regional, fire_regional, isop_dry_spatial, methanol_dry_spatial, hcho_dry_spatial, aod_dry_spatial, co_dry_spatial, no2_dry_spatial] 
# data_wet = [broadleaf_regional*100, lai_regional, fire_regional, isop_wet_spatial, methanol_wet_spatial, hcho_wet_spatial, aod_wet_spatial, co_wet_spatial, no2_wet_spatial] 

years = [np.arange(2001, 2020), np.arange(2001, 2020), np.arange(2001, 2020), np.arange(2001, 2017), \
         np.arange(2012, 2021), np.arange(2008, 2019), np.arange(2005, 2019),\
         np.arange(2001, 2020), np.arange(2005, 2021), np.arange(2001, 2020)]
labels = ['Broadleaf\nForest %', 'Savanna and\nGrassland %', 'LAI', 'Burned\nArea km$^{2}$', \
          'Isoprene\n10$^{15}$ mol cm$^{-2}$', 'Methanol\nppbv', 'HCHO\n10$^{16}$ mol cm$^{-2}$', \
              'CO\n10$^{17}$ mol cm$^{-2}$', 'NO$_{2}$\n10$^{15}$ mol cm$^{-2}$', 'AOD']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
cm =  1/2.54
fontsize = 8
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(12*cm,16*cm))
ax = ax.ravel()
for i in range(10):
    ax[i].plot(years[i], data[i], label = f'{labels[i]}') 
    ax[i].fill_between(years[i], data[i]-errors[i], \
                      data[i]+errors[i], alpha = 0.3)
    ax[i].set_ylabel(f'{labels[i]}', fontsize = fontsize)
    ax[i].set_title(f'({alphabet[i]})', loc = 'left', fontsize = fontsize)
    # ax[i].set_xlabel('Year', fontsize = 12)
    ax[i].set_xlim(2001, 2020)
    ax[i].set_xticks(year_ticks)
    ax[i].tick_params(axis='both', which='major', labelsize=fontsize)
    if i ==8 or i == 9:
        ax[i].set_xlabel('Year', fontsize = fontsize)

# fig.suptitle('Wet season')
fig.tight_layout()
# year_ticks = np.arange(2001, 2020, 2)
# ax.set_xticks(year_ticks)
# ax.set_ylim(-200, 400)
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/annual_through_time.png', dpi = 300)
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/annual_through_time_stderr.pdf')
# 