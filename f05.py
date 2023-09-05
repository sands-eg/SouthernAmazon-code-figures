# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:42:04 2023

@author: s2261807

Fire seasonal cycle
"""
### import functions
import xarray as xr
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt
from sklearn.linear_model import TheilSenRegressor
# import scipy.stats

# =============================================================================
# load data
# =============================================================================

# forest cover
fn = 'R:\modis_lc\mcd12c1_1deg_2001-2019_forest_prop_corrected.nc'
ds = xr.open_dataset(fn, decode_times=True)
fc = ds['fc'][:]
ds.close()

# cell areas
cell_areas_1 = np.zeros((180, 360))
cell_areas_1 = np.load('R:\modis_1_deg_cell_area.npy')
cell_areas_1xr = xr.DataArray(data = cell_areas_1[:], coords = {"lat": fc.lat.values, "lon":fc.lon.values})

# burned area
fn = 'R:\\gfed\\monthly_1degree_sum_2001-2016.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire = ds['Burned area']
fire['time'] = pd.date_range('2001-01-01', '2016-12-31', freq = 'MS')
ds.close()

# isoprene
fn = 'R:/cris_isoprene/2012-2020_CrIS_monthly_Isoprene_1degree_vAug23_2SDmask.nc'
ds = xr.open_dataset(fn, decode_times=True)
isop = ds['isoprene'][:-12,:,:]
ds.close()

# methanol
fn = 'R:/methanol/methanol_1degree_2008-2018.nc'
ds = xr.open_dataset(fn, decode_times=True)
methanol = ds['methanol'][:,:,:]
ds.close()

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
no2 = ds['no2'][:-12,:,:]
ds.close()

# AOD
fn = 'R:\modis_aod\mod08_aod_masked_2001-2019.nc'
ds = xr.open_dataset(fn, decode_times=True)
aod = ds['aod'][:,0,:,:]
# # mask_value = aod[0,0,0,0]
ds.close()

## DEM
fn = 'R:/DEM/SAM_DEM.nc'
ds = xr.open_dataset(fn)
dem = ds['SAM_DEM']
ds.close()  

# =============================================================================
# Define functions
# =============================================================================
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

def crop_ref_sector(var):
    ''' Function to crop data to area of interest:
        Pacific Reference Sector'''
    mask_lon = (var.lon >= -140) & (var.lon <= -100)
    mask_lat = (var.lat >= -30) & (var.lat <= 0)
    var_crop = var.where(mask_lon & mask_lat, drop=True)
    return var_crop

def spatial_weighted_average(data, weights):
    length = data.shape[0]
    weighted = np.ones(length)
    for i in range(length):
        weighted[i] = np.ma.average(np.ma.masked_invalid(data[i,:,:]), weights = weights)
    weighted_xr = xr.DataArray(data = weighted, coords = {"time": data.time})
    return weighted_xr

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

# =============================================================================
# Crop data
# =============================================================================

fire_crop = crop_data(fire)

hcho_crop = crop_data(hcho)
co_crop = crop_data(co)
no2_crop = crop_data(no2)
aod_crop = crop_data(aod)
isop_crop = crop_data(isop)
methanol_crop = crop_data(methanol)
dem_crop = crop_data(dem)

weights_crop = crop_data(spatial_weights)
# =============================================================================
# Detrend HCHO
# =============================================================================

pacific = crop_ref_sector(hcho)
pac_weights = crop_ref_sector(spatial_weights)

pacific_mean = spatial_weighted_average(pacific, pac_weights)


X = [i for i in range(0, len(pacific_mean))]
X = np.reshape(X, (len(X), 1))
y1 = pacific_mean
# Theil-Sen
model = TheilSenRegressor()
reg = model.fit(X, y1)
trend_TS = model.predict(X)
R2_ts = reg.score(X, y1)

trend_diff = trend_TS - trend_TS[0]

detrended_hcho = np.zeros_like(hcho_crop)
for i in range(len(hcho_crop)):
    detrended_hcho[i,:,:] = hcho_crop[i,:,:] - trend_diff[i]
        
hcho_crop = xr.DataArray(data = detrended_hcho, coords = {"time": hcho_crop.time, "lat": hcho_crop.lat, "lon": hcho_crop.lon})


# =============================================================================
# Mask high elevation
# =============================================================================

elev_boundary = 1000
high_elev = ma.masked_greater_equal(dem_crop, elev_boundary).mask


fire_elev = mask_high_elev(fire_crop, high_elev)

hcho_elev = mask_high_elev(hcho_crop, high_elev)
co_elev = mask_high_elev(co_crop, high_elev)
no2_elev = mask_high_elev(no2_crop, high_elev)
aod_elev = mask_high_elev(aod_crop, high_elev)
isop_elev = mask_high_elev(isop_crop, high_elev)
methanol_elev = mask_high_elev(methanol_crop, high_elev)

# =============================================================================
# Get monthly means
# =============================================================================

hcho_spatial = spatial_weighted_average(hcho_elev, weights_crop)
co_spatial = spatial_weighted_average(co_elev, weights_crop)
aod_spatial = spatial_weighted_average(aod_elev, weights_crop)
isop_spatial = spatial_weighted_average(isop_elev, weights_crop)
methanol_spatial = spatial_weighted_average(methanol_elev, weights_crop)
no2_spatial = spatial_weighted_average(no2_elev, weights_crop)
fire_spatial = spatial_weighted_average(fire_elev, weights_crop)

# =============================================================================
# Seasonal cycle
# =============================================================================

def seasonal_cycle(data):
    months = np.arange(1, 13)
    mon = data.groupby('time.month').groups
    seasonal_cycle = np.zeros(12) #(12, fire_crop.shape[1], fire_crop.shape[2]))
    for i, m in enumerate(months):
        mon_idxs = mon[m]
        seasonal_cycle[i]=data.isel(time=mon_idxs).mean()
        # option to add standard deviation etc
    return seasonal_cycle
        
hcho_season = seasonal_cycle(hcho_spatial)
fire_season = seasonal_cycle(fire_spatial)
isop_season = seasonal_cycle(isop_spatial)
co_season = seasonal_cycle(co_spatial)
aod_season = seasonal_cycle(aod_spatial)
no2_season = seasonal_cycle(no2_spatial)
methanol_season = seasonal_cycle(methanol_spatial)
    

data = [isop_season, methanol_season, hcho_season, aod_season, co_season, no2_season, fire_season]
months = np.arange(1, 13)
labels = ['Isoprene', 'Methanol', 'HCHO', 'AOD', 'CO', 'NO$_{2}$', 'Fire']

fig1, ax1 = plt.subplots(figsize=(12, 8))
for i in range(7):
    ax1.plot(months, data[i]/data[i].max(), label = f'{labels[i]}')
    print(data[i].max())
ax1.legend(loc='upper left')
ax1.set_ylabel('Proportion of maximum monthly mean', fontsize = 16)
ax1.set_xlabel('Month', fontsize = 16)


# fig1, ax1 = plt.subplots(figsize=(12, 8))
# ax1.plot(months, np.nanmean(seasonal_cycle, axis=(1,2)), label = 'methanol')
# ax2 = plt.twinx()
# ax2.plot(months, seasonal_cycle_fire*100, color = 'red', linestyle = ':', label = 'burned area')
# ax1.set_ylabel(f'{data_label}', fontsize = 16)
# ax2.set_ylabel('Burned area [% area]', fontsize = 16)
# ax1.set_xlabel('Month', fontsize = 16)
# ax1.set_title(f'{loc_label}')
# ax1.legend(loc='upper left')
# ax2.legend()

# save figure
# fig1.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/f05.png', dpi = 300)


# =============================================================================
# Figure version 2
# =============================================================================

data = [isop_season, methanol_season, hcho_season, aod_season, co_season, no2_season, fire_season]
months = np.arange(1, 13)
labels = ['Isoprene', 'Methanol', 'HCHO', 'AOD', 'CO', 'NO$_{2}$', 'Fire']

#calculate annual means - should create weighted version if using for manuscript
def calc_ann_mean(seasonal_cycle):
    ann_mean = seasonal_cycle.mean()
    return ann_mean

fig1, ax1 = plt.subplots(figsize=(12, 8))
for i in range(7):
    ax1.plot(months, (data[i]-calc_ann_mean(data[i]))/calc_ann_mean(data[i])*100, label = f'{labels[i]}')
    print(data[i].max())
ax1.legend(loc='upper left')
ax1.set_ylabel('% anomaly from annual mean', fontsize = 16)
ax1.set_xlabel('Month', fontsize = 16)


# # save figure
# fig1.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/f05_optionB_all.png', dpi = 300)

fig1, ax1 = plt.subplots(figsize=(12, 8))
for i in [0, 2, 4]:
    ax1.plot(months, (data[i]-calc_ann_mean(data[i]))/calc_ann_mean(data[i])*100, label = f'{labels[i]}')
    print(data[i].max())
ax1.plot(months, np.zeros((12)), c = 'grey', ls = ':')
ax1.legend(loc='upper left')
ax1.set_ylabel('% anomaly from annual mean', fontsize = 16)
ax1.set_xlabel('Month', fontsize = 16)


# # save figure
# fig1.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/f05_optionB_small.png', dpi = 300)

fig1, ax1 = plt.subplots(figsize=(12, 8))
for i in [1, 3, 5, 6]:
    ax1.plot(months, (data[i]-calc_ann_mean(data[i]))/calc_ann_mean(data[i])*100, label = f'{labels[i]}')
    print(data[i].max())
ax1.plot(months, np.zeros((12)), c = 'grey', ls = ':')
ax1.legend(loc='upper left')
ax1.set_ylabel('% anomaly from annual mean', fontsize = 16)
ax1.set_xlabel('Month', fontsize = 16)


# # save figure
# fig1.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/f05_optionB_large.png', dpi = 300)
