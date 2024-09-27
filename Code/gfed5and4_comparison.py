# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:50:25 2023

@author: s2261807

Burned area comparison plot
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
# fn = 'R:\\gfed\\monthly_1degree_sum_2001-2016.nc'
# ds = xr.open_dataset(fn, decode_times=True)
# fire = ds['Burned area'].fillna(0)*100 # to get % instead of fraction
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
fire2 = ds['__xarray_dataarray_variable__'].fillna(0)#*100 
fire2['time'] = pd.date_range('2001-01-01', '2020-12-31', freq = 'MS')

ds.close()


# =============================================================================
# Define functions
# =============================================================================
def crop_data(var):
    ''' Function to crop data to area of interest:
        South America ()
        50-70 degrees W, 5-25 degrees S'''
    ### Southern Amazon
    mask_lon = (var.lon >= -80) & (var.lon <= -30)
    mask_lat = (var.lat >= -80) & (var.lat <= 0)
    
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

fire2 = fire2.fillna(0)/surface_area_earth *100
fire = fire.fillna(0)/surface_area_earth *100
# fire = fire * surface_area_earth / 100
# =============================================================================
# Crop data
# =============================================================================
fire2_crop = crop_data(fire2)
fire_crop = crop_data(fire)

weights_crop = crop_data(spatial_weights)
fire_spatial = spatial_weighted_average(fire_crop, weights_crop)
fire2_spatial = spatial_weighted_average(fire2_crop, weights_crop)

mean_gfed4 = fire_spatial.mean()
mean_gfed5 = fire2_spatial[:-48].mean()


def seasonal_cycle(data):
    months = np.arange(1, 13)
    mon = data.groupby('time.month').groups
    seasonal_cycle = np.zeros((12,3)) #(12, fire_crop.shape[1], fire_crop.shape[2]))
    for i, m in enumerate(months):
        mon_idxs = mon[m]
        seasonal_cycle[i, 0]=data.isel(time=mon_idxs).mean()
        seasonal_cycle[i, 1]=data.isel(time=mon_idxs).std()
        seasonal_cycle[i, 2]=len(data.isel(time=mon_idxs))
        # option to add standard deviation etc
    return seasonal_cycle

season_gfed4 = seasonal_cycle(fire_spatial)
season_gfed5 = seasonal_cycle(fire2_spatial[:-48])



plt.plot(np.arange(1,13), season_gfed4[:,0])
plt.plot(np.arange(1,13), season_gfed5[:,0])
