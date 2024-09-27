# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:50:50 2024


Script for figure illustrating similarities between isoprene (NO2) and broadleaf forest (burned area)
and formaldehyde.

@author: s2261807
"""

# =============================================================================
# import functions and load data
# =============================================================================
#import packages and load data
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import pandas as pd
from sklearn.linear_model import TheilSenRegressor
from matplotlib_scalebar.scalebar import ScaleBar
from math import radians, sin, cos, acos
import matplotlib as mpl


# hcho
fn = 'R:/OMI_HCHO/OMI_HCHO_RSCVC_monthly_no_corrections.nc'
ds = xr.open_dataset(fn, decode_times=True)
hcho = ds['hcho_rs'][:-12,::-1,:]
ds.close()

# isoprene
fn = 'R:/cris_isoprene/2012-2020_CrIS_monthly_Isoprene_1degree_vAug23_2SDmask.nc'
ds = xr.open_dataset(fn, decode_times=True)
isop = ds['isoprene'][:-12,:,:]
ds.close()

# # NO2
fn = 'R:/OMI_NO2/omi_no2_mm_2005_2020_masked.nc'
ds = xr.open_dataset(fn, decode_times=True)
# ds['time'] = pd.date_range('2005-01-01', '2020-12-31', freq = 'MS')
no2 = ds['no2'][:-12,:,:]
ds.close()

# burned area (GFED 4; 2001-2016); error in metadata - unit is fraction of cell, not %
fn = 'R:\\gfed\\GFED4_BAkm2_2001-2016.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire = ds['Burned Area']
fire['time'] = pd.date_range('2001-01-01', '2016-12-31', freq = 'MS')
ds.close()

fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=True)
broadleaf = (ds['Land_Cover_Type_1_Percent'][:,:,:,2] + ds['Land_Cover_Type_1_Percent'][:,:,:,4])
broadleaf['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()

## DEM
fn = 'R:/DEM/SAM_DEM.nc'
ds = xr.open_dataset(fn)
dem = ds['SAM_DEM']
ds.close()  
# =============================================================================
# define additional functions
# =============================================================================
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

def crop_data(var):
    ''' Function to crop data to area of interest:
        South America ()
        50-70 degrees W, 5-25 degrees S'''    
    ### Southern Amazon
    mask_lon = (var.lon >= -70) & (var.lon <= -50)
    mask_lat = (var.lat >= -25) & (var.lat <= -5)
    var_crop = var.where(mask_lon & mask_lat, drop=True)
    return var_crop

# =============================================================================
# calculate mean land surface fractions
# =============================================================================

elev = crop_data(dem)
high_elev = elev.where(elev >= 1000)

#### in km^2
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
for i in range(n_lons):
    for j in range(n_lats):
        term1 = R**2
        term2 = (lons_rad[i+1] - lons_rad[i])
        term3 = np.sin(lats_rad[j+1]) - np.sin(lats_rad[j])
        
        tmp_sa = -1*term1*term2*term3 # without -1 surface area comes out negative (lat in different order for MODIS?)
        surface_area_earth[j, i] = tmp_sa
cell_areas = xr.DataArray(data = surface_area_earth, coords = {"lat": fire.lat, "lon": fire.lon})

crop_areas = crop_data(cell_areas)

fire = fire.fillna(0)/cell_areas *100

# =============================================================================
# detrend hcho
# =============================================================================
hcho_wet = get_month_data([2,3,4], hcho)
hcho_dry = get_month_data([8,9,10], hcho)
hcho_dry_mean_orig = weighted_temporal_mean(hcho_dry)
hcho_wet_mean_orig = weighted_temporal_mean(hcho_wet)

def crop_ref_sector(var):
    ''' Function to crop data to area of interest:
        Pacific reference sector'''
    mask_lon = (var.lon >= -140) & (var.lon <= -100)
    mask_lat = (var.lat >= -30) & (var.lat <= 0)
    var_crop = var.where(mask_lon & mask_lat, drop=True)
    return var_crop

pacific_wet = crop_ref_sector(hcho_wet_mean_orig)
pacific_dry = crop_ref_sector(hcho_dry_mean_orig)

pacific_mean_wet = pacific_wet.mean(axis=(1,2))
pacific_mean_dry = pacific_dry.mean(axis=(1,2))

X = [i for i in range(0, len(pacific_mean_wet))]
X = np.reshape(X, (len(X), 1))
y1 = pacific_mean_wet.values #

# Theil-Sen
model = TheilSenRegressor()
reg = model.fit(X, y1)
trend_TS_wet = model.predict(X)
R2_ts_wet = reg.score(X, y1)

X = [i for i in range(0, len(pacific_mean_dry))]
X = np.reshape(X, (len(X), 1))
y2 = pacific_mean_dry.values #

# Theil-Sen
model = TheilSenRegressor()
reg = model.fit(X, y2)
trend_TS_dry = model.predict(X)
R2_ts_dry = reg.score(X, y2)

trend_diff_wet = trend_TS_wet - trend_TS_wet[0]
trend_diff_dry = trend_TS_dry - trend_TS_dry[0]

detrended_hcho_wet = np.zeros_like(hcho_wet)
for i, y in enumerate(range(14)):
    for j in range(3):
        detrended_hcho_wet[y*3+j,:,:] = hcho_wet[y*3+j,:,:] - trend_diff_wet[i]
        
hcho_wet_d = xr.DataArray(data = detrended_hcho_wet, coords = {"time": hcho_wet.time, "lat": hcho_wet.lat, "lon": hcho_wet.lon})

detrended_hcho_dry = np.zeros_like(hcho_dry)
for i, y in enumerate(range(14)):
    for j in range(3):
        detrended_hcho_dry[y*3+j,:,:] = hcho_dry[y*3+j,:,:] - trend_diff_dry[i]

hcho_dry_d = xr.DataArray(data = detrended_hcho_dry, coords = {"time": hcho_dry.time, "lat": hcho_dry.lat, "lon": hcho_dry.lon})

# =============================================================================
# calculate mean atmospheric composition for wet and dry seasons
# =============================================================================

isop_wet = get_month_data([2,3,4], isop)
isop_dry = get_month_data([8,9,10], isop)
isop_dry_mean = crop_data(weighted_temporal_mean(isop_dry).mean(axis=0))
isop_wet_mean = crop_data(weighted_temporal_mean(isop_wet).mean(axis=0))

no2_wet = get_month_data([2,3,4], no2)
no2_dry = get_month_data([8,9,10], no2)
no2_dry_mean = crop_data(weighted_temporal_mean(no2_dry).mean(axis=0))
no2_wet_mean = crop_data(weighted_temporal_mean(no2_wet).mean(axis=0))

hcho_dry_mean = crop_data(weighted_temporal_mean(hcho_dry_d).mean(axis=0))
hcho_wet_mean = crop_data(weighted_temporal_mean(hcho_wet_d).mean(axis=0))

# fire_wet = get_month_data([2,3,4], fire)
fire_dry = get_month_data([8,9,10], fire)
fire_dry_mean = crop_data(weighted_temporal_mean(fire_dry).mean(axis=0))
# fire_wet_mean = crop_data(weighted_temporal_mean(fire_wet).mean(axis=0))
# =============================================================================
# get broadleaf forest mean
# =============================================================================
forest_mean = broadleaf.mean(axis=0)
forest_crop = crop_data(forest_mean)

# =============================================================================
# maps 
# =============================================================================
min_lon = -70
max_lon =  -50 
min_lat =   -25  
max_lat = -5 

projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = isop_dry_mean.lon
latitude = isop_wet_mean.lat

data = [forest_crop, isop_dry_mean/10**16]#, forest_crop, hcho_dry_mean/10**16]
#, hcho_dry_mean, hcho_wet_mean,\
#        co_dry_mean, co_wet_mean, aod_dry_mean, aod_wet_mean,\
#            no2_dry_mean, no2_wet_mean, methanol_dry_mean, methanol_wet_mean]

# set display parameters
vmin1 = 0
vmax1 = 2
scaler1 = 10**16
cmap1 = plt.cm.get_cmap('YlOrRd')
levels1 = np.linspace(vmin1*scaler1, vmax1*scaler1, 11)
cmap1 = plt.cm.get_cmap('YlOrRd')
# extract all colors from the  map
cmaplist = [cmap1(i) for i in range(cmap1.N)]
# create the new map
cmap1a = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap1.N)
# define the bins and normalize
bounds = np.linspace(0, 2, 11)
norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)

cmap2 = plt.cm.get_cmap('gist_earth_r')
# extract all colors from the  map
cmaplist = [cmap2(i) for i in range(cmap2.N)]
# create the new map
cmap2a = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap2.N)

# define the bins and normalize
bounds = np.linspace(0, 100, 11)
norm2 = mpl.colors.BoundaryNorm(bounds, cmap2.N)

vmin2 = 0
vmax2 = 1000
scaler2 = 1
cmap2 = plt.cm.get_cmap('Greys')
levels2 = np.linspace(vmin2*scaler2, vmax2*scaler2, 11)

# set up figure
cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30*cm, 45*cm), subplot_kw={'projection':projection})
axes = axes.ravel()

# first panel (saved as im for colour bar creation)
im1 = axes[0].pcolormesh(longitude, latitude, data[0], cmap = cmap2a, norm = norm2, transform=transform)
im2 = axes[1].pcolormesh(longitude, latitude, data[1], cmap = cmap1a, norm = norm, transform=transform)

# repeat for other years
for i,y in enumerate(data):
    axes[i].coastlines()
    axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    axes[i].contourf(longitude, latitude, high_elev, levels = levels2, cmap = cmap2, extend = 'max', alpha = 0.7, transform=transform )
    # axes[i].contour(longitude, latitude, forest_crop, levels = np.linspace(0.51, 1, 2), cmap='Blues_r', \
    #                 alpha = 0.5, transform=transform )
    gl = axes[i].gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-65, -55])
    gl.ylocator = mticker.FixedLocator([-10, -20])
    gl.xlabel_style = {'size' : 26}
    gl.ylabel_style = {'size' : 26}
    axes[i].add_feature(cfeature.BORDERS, zorder=10)


# scalebar = ScaleBar(dx, box_alpha=0.6)#, location = 'lower right', box_color='white') #units = 'deg', dimension = 'angle')
# plt.gca().add_artist(scalebar)
    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
fig.subplots_adjust(right=0.9)

# add colorbars
cax1 = fig.add_axes([0.85, 0.55, 0.02, 0.4])
cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
cb1.ax.tick_params(labelsize=26)
cb1.set_label('Broadleaf forest cover (%)', weight = 'bold', fontsize = 35)

cax2 = fig.add_axes([0.85, 0.05, 0.02, 0.4])
cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
cb2.ax.tick_params(labelsize=26)
cb2.set_label('Isoprene (10$^{16}$ molecules cm$^{-2}$)', weight = 'bold', fontsize = 35)

# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/biogenic_EGU.png', dpi = 300)

# =============================================================================
# burned area and NO2
# =============================================================================

min_lon = -70
max_lon =  -50 
min_lat =   -25  
max_lat = -5 

projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = isop_dry_mean.lon
latitude = isop_wet_mean.lat

data = [fire_dry_mean, no2_dry_mean/10**15]#, forest_crop, hcho_dry_mean/10**16]
#, hcho_dry_mean, hcho_wet_mean,\
#        co_dry_mean, co_wet_mean, aod_dry_mean, aod_wet_mean,\
#            no2_dry_mean, no2_wet_mean, methanol_dry_mean, methanol_wet_mean]

# set display parameters
vmin1 = 0
vmax1 = 3
scaler1 = 10**15
cmap1 = plt.cm.get_cmap('YlOrRd')
levels1 = np.linspace(vmin1*scaler1, vmax1*scaler1, 16)
cmap1 = plt.cm.get_cmap('YlOrRd')
# extract all colors from the  map
cmaplist = [cmap1(i) for i in range(cmap1.N)]
# create the new map
cmap1a = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap1.N)
# define the bins and normalize
bounds = np.linspace(0, 3, 16)
norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)

cmap2a = plt.cm.get_cmap('magma_r')
# # extract all colors from the  map
# cmaplist = [cmap2(i) for i in range(cmap2.N)]
# # create the new map
# cmap2a = mpl.colors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap2.N)

# # define the bins and normalize
# bounds = np.linspace(0, 0.5, 21)
# norm2 = mpl.colors.BoundaryNorm(bounds, cmap2.N)

vmin2 = 0
vmax2 = 1000
scaler2 = 1
cmap2 = plt.cm.get_cmap('Greys')
levels2 = np.linspace(vmin2*scaler2, vmax2*scaler2, 11)

# set up figure
cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30*cm, 45*cm), subplot_kw={'projection':projection})
axes = axes.ravel()

# first panel (saved as im for colour bar creation)
im1 = axes[0].pcolormesh(longitude, latitude, data[0], cmap = cmap2a, vmin = 0, vmax = 0.3, transform=transform)
im2 = axes[1].pcolormesh(longitude, latitude, data[1], cmap = cmap1a, norm = norm, transform=transform)

# repeat for other years
for i,y in enumerate(data):
    axes[i].coastlines()
    axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    axes[i].contourf(longitude, latitude, high_elev, levels = levels2, cmap = cmap2, extend = 'max', alpha = 0.7, transform=transform )
    # axes[i].contour(longitude, latitude, forest_crop, levels = np.linspace(0.51, 1, 2), cmap='Blues_r', \
    #                 alpha = 0.5, transform=transform )
    gl = axes[i].gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-65, -55])
    gl.ylocator = mticker.FixedLocator([-10, -20])
    gl.xlabel_style = {'size' : 26}
    gl.ylabel_style = {'size' : 26}
    axes[i].add_feature(cfeature.BORDERS, zorder=10)


# scalebar = ScaleBar(dx, box_alpha=0.6)#, location = 'lower right', box_color='white') #units = 'deg', dimension = 'angle')
# plt.gca().add_artist(scalebar)
    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
fig.subplots_adjust(right=0.9)

# add colorbars
cax1 = fig.add_axes([0.85, 0.55, 0.02, 0.4])
cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
cb1.ax.tick_params(labelsize=26)
cb1.set_label('Burned area (%)', weight = 'bold', fontsize = 35)

cax2 = fig.add_axes([0.85, 0.05, 0.02, 0.4])
cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
cb2.ax.tick_params(labelsize=26)
cb2.set_label('Nitrogen dioxide (10$^{15}$ mol cm$^{-2}$)', weight = 'bold', fontsize = 35)

# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/pyrogenic_EGU.png', dpi = 300)


# =============================================================================
# formaldehyde
# =============================================================================

min_lon = -70
max_lon =  -50 
min_lat =   -25  
max_lat = -5 

projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = isop_dry_mean.lon
latitude = isop_wet_mean.lat

data = [hcho_dry_mean/10**16]#, forest_crop, hcho_dry_mean/10**16]
#, hcho_dry_mean, hcho_wet_mean,\
#        co_dry_mean, co_wet_mean, aod_dry_mean, aod_wet_mean,\
#            no2_dry_mean, no2_wet_mean, methanol_dry_mean, methanol_wet_mean]

# set display parameters
vmin1 = 0
vmax1 = 2
scaler1 = 10**16
cmap1 = plt.cm.get_cmap('YlOrRd')
levels1 = np.linspace(vmin1*scaler1, vmax1*scaler1, 21)
cmap1 = plt.cm.get_cmap('YlOrRd')
# extract all colors from the  map
cmaplist = [cmap1(i) for i in range(cmap1.N)]
# create the new map
cmap1a = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap1.N)
# define the bins and normalize
bounds = np.linspace(0, 2, 21)
norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)


vmin2 = 0
vmax2 = 1000
scaler2 = 1
cmap2 = plt.cm.get_cmap('Greys')
levels2 = np.linspace(vmin2*scaler2, vmax2*scaler2, 11)

# set up figure
cm = 1/2.54
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 23*cm), subplot_kw={'projection':projection})

# first panel (saved as im for colour bar creation)
im1 = axes.pcolormesh(longitude, latitude, data[0], cmap = cmap1a, norm = norm, transform=transform)

# repeat for other years
axes.coastlines()
axes.set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
axes.contourf(longitude, latitude, high_elev, levels = levels2, cmap = cmap2, extend = 'max', alpha = 0.7, transform=transform )
gl = axes.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator([-65, -55])
gl.ylocator = mticker.FixedLocator([-10, -20])
gl.xlabel_style = {'size' : 26}
gl.ylabel_style = {'size' : 26}
axes.add_feature(cfeature.BORDERS, zorder=10)


# scalebar = ScaleBar(dx, box_alpha=0.6)#, location = 'lower right', box_color='white') #units = 'deg', dimension = 'angle')
# plt.gca().add_artist(scalebar)
    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
fig.subplots_adjust(right=0.9)

# add colorbars
cax1 = fig.add_axes([0.85, 0.1, 0.02, 0.8])
cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
cb1.ax.tick_params(labelsize=26)
cb1.set_label('Formaldehyde (10$^{15}$ mol cm$^{-2}$)', weight = 'bold', fontsize = 35)


# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/hcho_EGU.png', dpi = 300)
