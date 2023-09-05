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


### data

## land cover, burned area

# ### LC option A
# broadleaf forest
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


# ## LC option B
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

# fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
# ds = xr.open_dataset(fn, decode_times=False)
# grass_crop = ds['Land_Cover_Type_1_Percent'][:,:,:,10] + ds['Land_Cover_Type_1_Percent'][:,:,:,12]
# ds.close()

fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=False)
grass_mosaic = (ds['Land_Cover_Type_1_Percent'][:,:,:,10] + ds['Land_Cover_Type_1_Percent'][:,:,:,12] + ds['Land_Cover_Type_1_Percent'][:,:,:,14])/100
grass_mosaic['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()
grass = grass_mosaic


# burned area (GFED 4; 2001-2016); error in metadata - unit is fraction of cell, not %
fn = 'R:\\gfed\\monthly_1degree_sum_2001-2016.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire = ds['Burned area']*100 
fire['time'] = pd.date_range('2001-01-01', '2016-12-31', freq = 'MS')
ds.close()

## DEM
fn = 'R:/DEM/SAM_DEM.nc'
ds = xr.open_dataset(fn)
dem = ds['SAM_DEM']
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

# =============================================================================
# Define additional functions
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

# =============================================================================
# Crop data
# =============================================================================
fire_crop = crop_data(fire)

broadleaf_crop = crop_data(broadleaf)
grass_crop = crop_data(grass)
savanna_crop = crop_data(savanna)

dem_crop = crop_data(dem)
weights_crop = crop_data(spatial_weights)
# =============================================================================
# mask high altitude areas
# =============================================================================
elev_boundary = 1000
high_elev = ma.masked_greater_equal(dem_crop, elev_boundary).mask

broadleaf_mask = mask_high_elev(broadleaf_crop, high_elev) 
grass_mask = mask_high_elev(grass_crop, high_elev) 
savanna_mask = mask_high_elev(savanna_crop, high_elev) 
fire_mask = mask_high_elev(fire_crop, high_elev) * crop_data(surface_area)

# =============================================================================
# Get annual fire sum
# =============================================================================
fire_sum = fire_mask.groupby("time.year").sum()

# =============================================================================
# Regional means (sum in case of fires)
# =============================================================================
broadleaf_regional = spatial_weighted_average(broadleaf_mask, weights_crop)
grass_regional = spatial_weighted_average(grass_mask, weights_crop)
savanna_regional = spatial_weighted_average(savanna_mask, weights_crop)
fire_regional = fire_sum.sum(axis=(1,2))

# =============================================================================
# Plot
# =============================================================================
year = np.arange(2001, 2020)

fig, ax = plt.subplots(figsize=(13,9))

ax.plot(year, broadleaf_regional*100, color = 'blue', label = 'Broadleaf')
ax2 = ax.twinx()
ax2.plot(year, grass_regional*100, color = 'orange', label = 'Grasslands')
ax2.plot(year, savanna_regional*100, color = 'maroon', label = 'Savanna')

year_ticks = np.arange(2001, 2020,2)
ax.set_xticks(year_ticks)
ax.set_ylabel('Broadleaf [%]', fontsize = 16)
ax2.set_ylabel('Grassland or Savanna [%]', fontsize = 16)
ax.set_xlabel('Year', fontsize = 16)
fig.legend(fontsize = 14, loc = 'center right', bbox_to_anchor=(0.4, 0.3, 0.5, 0.5))


# save figure
# fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/MonteCarlo/{labels[atmos_no]}_{lc_label}_lineandbootstrap_noneabove{elev_boundary}.png', dpi = 300)


year = np.arange(2001, 2017)

fig, ax = plt.subplots(figsize=(13,9))

ax.plot(year, fire_regional, color = 'maroon', label = 'Broadleaf')

year_ticks = np.arange(2001, 2017,2)
ax.set_xticks(year_ticks)
ax.set_ylabel('Burned Area', fontsize = 16)
ax.set_xlabel('Year', fontsize = 16)
# fig.legend(fontsize = 14, loc = 'center right', bbox_to_anchor=(0.4, 0.3, 0.5, 0.5))