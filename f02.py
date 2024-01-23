# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:30:14 2023

@author: s2261807

Script for creating regional maps over the Southern Amazon:
    - mean percentage cover of forests, grasslands/pastures, shrubs
    - mean wet and dry season atmospheric composition

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
# from sklearn.linear_model import TheilSenRegressor
from matplotlib_scalebar.scalebar import ScaleBar
from math import radians, sin, cos, acos

# hcho
fn = 'R:/OMI_HCHO/OMI_HCHO_RSCVC_monthly_no_corrections.nc'
ds = xr.open_dataset(fn, decode_times=True)
hcho = ds['hcho_rs'][:-12,::-1,:]
ds.close()

# isoprene
fn = 'R:/cris_isoprene/2012-2020_CrIS_monthly_Isoprene_1degree_july2017interpolated.nc'
ds = xr.open_dataset(fn, decode_times=True)
isop = ds['isoprene'][:-12,:,:]
ds.close()

# # AOD
fn = 'R:\modis_aod\mod08_aod_masked_2001-2019.nc'
ds = xr.open_dataset(fn, decode_times=True)
aod = ds['aod'][:,0,:,:]
ds.close()

# CO
fn = 'R:/mopitt_co/mopitt_co_totalcolumn_2001-2019_monthly.nc'
ds = xr.open_dataset(fn, decode_times=True)
co = ds['__xarray_dataarray_variable__']
ds.close()

# # NO2
fn = 'R:/OMI_NO2/omi_no2_mm_2005_2020_masked.nc'
ds = xr.open_dataset(fn, decode_times=True)
# ds['time'] = pd.date_range('2005-01-01', '2020-12-31', freq = 'MS')
no2 = ds['no2'][:-12,:,:]
ds.close()

# methanol
fn = 'R:/methanol/methanol_1degree_2008-2018.nc'
ds = xr.open_dataset(fn, decode_times=True)
methanol = ds['methanol'][:,:,:]
ds.close()

# burned area (GFED 4; 2001-2016); error in metadata - unit is fraction of cell, not %
fn = 'R:\\gfed\\monthly_1degree_sum_2001-2016.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire = ds['Burned area']*100 
fire['time'] = pd.date_range('2001-01-01', '2016-12-31', freq = 'MS')
ds.close()

# burned area GFED5
fn = 'R:\gfed\GFED5\GFED5_totalBA_2001-2020.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire2 = ds['__xarray_dataarray_variable__'].fillna(0)
fire2['time'] = pd.date_range('2001-01-01', '2020-12-31', freq = 'MS')

ds.close()


# grid cell area
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
fire2 = fire2.fillna(0)/surface_area_earth *100
### land cover
# fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
# ds = xr.open_dataset(fn, decode_times=True)
# fc = ds['Land_Cover_Type_1_Percent'][:,:,:,1:6].sum(axis=3)
# ds.close()

# fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
# ds = xr.open_dataset(fn, decode_times=False)
# shrub = ds['Land_Cover_Type_1_Percent'][:,:,:,6:8].sum(axis=3)
# ds.close()

# fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
# ds = xr.open_dataset(fn, decode_times=False)
# savanna = ds['Land_Cover_Type_1_Percent'][:,:,:,8:10].sum(axis=3)
# ds.close()


# fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
# ds = xr.open_dataset(fn, decode_times=False)
# grass_crop = ds['Land_Cover_Type_1_Percent'][:,:,:,10] + ds['Land_Cover_Type_1_Percent'][:,:,:,12]
# ds.close()

# fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
# ds = xr.open_dataset(fn, decode_times=False)
# grass_mosaic = ds['Land_Cover_Type_1_Percent'][:,:,:,10] + ds['Land_Cover_Type_1_Percent'][:,:,:,12] + ds['Land_Cover_Type_1_Percent'][:,:,:,14]
# ds.close()

# fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
# ds = xr.open_dataset(fn, decode_times=False)
# urban = ds['Land_Cover_Type_1_Percent'][:,:,:,13]
# ds.close()

# broadleaf forest
fn = 'R:\modis_lc\mcd12c1_broadleaf_1deg.nc'
ds = xr.open_dataset(fn, decode_times=False)
broadleaf = ds['broadleaf'][:]
broadleaf['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()

# grasses and crops
fn = 'R:\modis_lc\mcd12c1_grass_crop_1deg.nc'
ds = xr.open_dataset(fn, decode_times=False)
grass = ds['grass_crop'][:]
grass['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()

# savanna
fn = 'R:\modis_lc\mcd12c1_1deg_igbp_savanna_majority.nc'
ds = xr.open_dataset(fn, decode_times=False)
savanna = ds['Savanna'][:]/100
savanna['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()


### LC option B
# fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
# ds = xr.open_dataset(fn, decode_times=True)
# fc = ds['Land_Cover_Type_1_Percent'][:,:,:,1:6].sum(axis=3)
# fc['time'] = pd.date_range('2001', '2020', freq = 'Y')
# ds.close()

fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=False)
grass_mosaic = (ds['Land_Cover_Type_1_Percent'][:,:,:,10] + ds['Land_Cover_Type_1_Percent'][:,:,:,12] + ds['Land_Cover_Type_1_Percent'][:,:,:,14])/100
grass_mosaic['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()

grass = grass_mosaic

fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=True)
broadleaf = (ds['Land_Cover_Type_1_Percent'][:,:,:,2] + ds['Land_Cover_Type_1_Percent'][:,:,:,4])/100
broadleaf['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()

# fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
# ds = xr.open_dataset(fn, decode_times=False)
# shrub = ds['Land_Cover_Type_1_Percent'][:,:,:,6:8].sum(axis=3)
# shrub['time'] = pd.date_range('2001', '2020', freq = 'Y')
# ds.close()

fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=False)
savanna = ds['Land_Cover_Type_1_Percent'][:,:,:,8:10].sum(axis=3)/100
savanna['time'] = pd.date_range('2001', '2020', freq = 'Y')
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

def great_circle(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6371 * 1000 * (
        acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))
    )
dx = great_circle(-51, -5, -50, -5)

# =============================================================================
# calculate mean land surface fractions
# =============================================================================

# fc_mean = crop_data(fc).mean(axis=0)
# grass_mean = crop_data(grass_crop).mean(axis=0)
# shrub_mean = crop_data(shrub).mean(axis=0)
# savanna_mean = crop_data(savanna).mean(axis=0)
# grass_mosaic_mean = crop_data(grass_mosaic).mean(axis=0)
# urban_mean = crop_data(urban).mean(axis=0)

broadleaf_mean = crop_data(broadleaf).mean(axis=0)
grass_mean = crop_data(grass).mean(axis=0)
savanna_mean = crop_data(savanna).mean(axis=0)
elev = crop_data(dem)
high_elev = elev.where(elev >= 1000)

grass_sav = grass_mean + savanna_mean

# =============================================================================
# display forest, grasslands and savanna and elevation maps
# =============================================================================
min_lon = -70
max_lon =  -50 
min_lat =   -25  
max_lat = -5 

# set projections and initialise data
projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = broadleaf_mean.lon
latitude = broadleaf_mean.lat

data = [grass_sav, broadleaf_mean, elev]
labels = ['a) grasslands and savanna', 'b) broadleaf forest', 'c) elevation']

# set display parameters
vmin = 0
vmax = 1
scaler = 100
cmap1 = plt.cm.get_cmap('Greens')
levels = np.linspace(vmin*scaler, vmax*scaler, 11)

vmin2 = 0
vmax2 = 1000
scaler2 = 1
cmap2 = plt.cm.get_cmap('Blues')
levels2 = np.linspace(vmin2*scaler2, vmax2*scaler2, 11)

vmin3 = 0
vmax3 = 5000
scaler3 = 1
cmap3 = plt.cm.get_cmap('gist_earth')
levels3 = np.linspace(vmin3*scaler3, vmax3*scaler3, 21)

# set up figure
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,6), subplot_kw={'projection':projection})
axes = axes.ravel()

# first panel (saved as im for colour bar creation)
im = axes[0].contourf(longitude, latitude, data[0]*scaler, levels = levels, cmap = cmap1, extend = 'neither', transform=transform)
im2 = axes[2].contourf(longitude, latitude, data[2]*scaler3, levels = levels3, cmap = cmap3, extend = 'both', transform=transform)

# repeat for other years
for i,y in enumerate(data):
    axes[i].set_title(labels[i], fontsize = 12)
    axes[i].coastlines()
    if i == 2:
        axes[i].contourf(longitude, latitude, data[i]*scaler3, levels = levels3, cmap = cmap3, extend = 'both', transform=transform)
    else:
        axes[i].contourf(longitude, latitude, data[i]*scaler, levels = levels, cmap = cmap1, extend = 'max', transform=transform)
        axes[i].contourf(longitude, latitude, high_elev, levels = levels2, cmap = cmap2, extend = 'max', alpha = 0.3, transform=transform )
    # axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    gl = axes[i].gridlines(draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-65, -55])
    gl.ylocator = mticker.FixedLocator([-10, -20])
    axes[i].add_feature(cfeature.BORDERS, zorder=10)

    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)

# add colorbar, vertical or horizontal
# cax = fig.add_axes([0.65, 0.05, 0.02, 0.87])
# cb = fig.colorbar(im, cax=cax, orientation='vertical')
cax = fig.add_axes([0.1, 0.2, 0.8, 0.02])
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cax2 = fig.add_axes([0.1, 0.1, 0.8, 0.02])
cb2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')


cb.set_label('% grid cell cover (subplots a, b)', fontsize = 12)
cb2.set_label('Elevation m a.s.l. (subplot c)', fontsize = 12)
# save figure
# fig.savefig('M:/figures/atm_chem/comparison/SouthAmazon_stats/Summary/landcover_and_elevation.png', dpi = 300)


# =============================================================================
# display 3 land cover maps
# =============================================================================
min_lon = -70
max_lon =  -50 
min_lat =   -25  
max_lat = -5 

# set projections and initialise data
projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = broadleaf_mean.lon
latitude = broadleaf_mean.lat

data = [grass_mean, savanna_mean, broadleaf_mean]
labels = ['Grasslands', 'Savanna', 'Broadleaf forest']

# set display parameters
vmin = 0
vmax = 1
scaler = 100
cmap1 = plt.cm.get_cmap('Greens')
levels = np.linspace(vmin*scaler, vmax*scaler, 11)

vmin2 = 0
vmax2 = 1000
scaler2 = 1
cmap2 = plt.cm.get_cmap('Blues')
levels2 = np.linspace(vmin2*scaler2, vmax2*scaler2, 11)

# set up figure
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,8), subplot_kw={'projection':projection})
axes = axes.ravel()

# first panel (saved as im for colour bar creation)
im = axes[0].contourf(longitude, latitude, data[0]*scaler, levels = levels, cmap = cmap1, extend = 'neither', transform=transform)

# repeat for other years
for i,y in enumerate(data):
    axes[i].set_title(labels[i], fontsize = 12)
    axes[i].coastlines()
    axes[i].contourf(longitude, latitude, data[i]*scaler, levels = levels, cmap = cmap1, extend = 'max', transform=transform)
    axes[i].contourf(longitude, latitude, high_elev, levels = levels2, cmap = cmap2, extend = 'max', alpha = 0.3, transform=transform )
    # axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    # gl = axes[i].gridlines(draw_labels=True)
    # gl.xlocator = mticker.FixedLocator([-75, -60, -45])
    # gl.ylocator = mticker.FixedLocator([0, -15, -30])

    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
# fig.subplots_adjust(top=0.92, right=0.9)

# add colorbar, vertical or horizontal
# cax = fig.add_axes([0.65, 0.05, 0.02, 0.87])
# cb = fig.colorbar(im, cax=cax, orientation='vertical')
cax = fig.add_axes([0.1, 0.2, 0.8, 0.02])
cb = fig.colorbar(im, cax=cax, orientation='horizontal')

cb.set_label('% grid cell cover', fontsize = 12)

# save figure
# fig.savefig('.png', dpi = 300)

# =============================================================================
# on one map?
# =============================================================================
# set projections and initialise data
projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = broadleaf_mean.lon
latitude = broadleaf_mean.lat

data = [grass_mean, savanna_mean, broadleaf_mean]
labels = ['Grasslands', 'Savanna', 'Broadleaf forest']

# set display parameters
vmin = 0.4
vmax = 0.9
scaler = 100
cmap1 = plt.cm.get_cmap('Oranges')
cmap2 = plt.cm.get_cmap('Purples')
cmap3 = plt.cm.get_cmap('Greens')
levels = np.linspace(vmin*scaler, vmax*scaler, 21)
alpha = 1
# set up figure
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(14,8), subplot_kw={'projection':projection})
# first panel (saved as im for colour bar creation)
im0 = axes.contourf(longitude, latitude, data[0]*scaler, levels = levels, cmap = cmap1, extend = 'max', alpha = alpha, transform=transform)
im1 =axes.contourf(longitude, latitude, data[1]*scaler, levels = levels, cmap = cmap2, extend = 'max', alpha = alpha, transform=transform)
im2 =axes.contourf(longitude, latitude, data[2]*scaler, levels = levels, cmap = cmap3, extend = 'max', alpha = alpha, transform=transform)

axes.coastlines()
axes.add_feature(cfeature.BORDERS, edgecolor = 'gray')
axes.set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
gl = axes.gridlines(draw_labels=True)
gl.xlocator = mticker.FixedLocator([-65, -60, -55])
gl.ylocator = mticker.FixedLocator([-10, -20])

scalebar = ScaleBar(dx) #units = 'deg', dimension = 'angle')
plt.gca().add_artist(scalebar)
    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
fig.subplots_adjust(top=0.9, right=0.8)

# add colorbar, vertical or horizontal
cax = fig.add_axes([0.7, 0.05, 0.02, 0.85])
cb = fig.colorbar(im0, cax=cax, orientation='vertical')
# cax = fig.add_axes([0.1, 0.0, 0.8, 0.02])
# cb = fig.colorbar(im0, cax=cax, orientation='horizontal')
cb.set_label(f'% {labels[0]}', fontsize = 12)

cax1 = fig.add_axes([0.8, 0.05, 0.02, 0.85])
cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
cb1.set_label(f'% {labels[1]}', fontsize = 12)

cax2 = fig.add_axes([0.9, 0.05, 0.02, 0.85])
cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
cb2.set_label(f'% {labels[2]}', fontsize = 12)

# save figure
# fig.savefig('.png', dpi = 300)

# =============================================================================
# Mean monthly fire for wet and dry season
# =============================================================================
# min_time = xr.DataArray(data = [fire.time.min().values, isop.time.min().values]).max().values
# min_year = min_time.astype(str).split('-')[0]
# max_time = xr.DataArray(data = [fire.time.max().values, isop.time.max().values]).min().values
# max_year = max_time.astype(str).split('-')[0]

# fire_slice = fire.sel(time=slice(min_year, max_year))
# fire_wet = get_month_data([2,3,4], fire_slice)
# fire_dry = get_month_data([8,9,10], fire_slice) 

# fire_wet = get_month_data([2,3,4], fire)
# fire_dry = get_month_data([8,9,10], fire)
# fire_dry_mean = crop_data(weighted_temporal_mean(fire_dry).mean(axis=0))
# fire_wet_mean = crop_data(weighted_temporal_mean(fire_wet).mean(axis=0))

fire_wet = get_month_data([2,3,4], fire2)
fire_dry = get_month_data([8,9,10], fire2)
fire_dry_mean = crop_data(weighted_temporal_mean(fire_dry).mean(axis=0))
fire_wet_mean = crop_data(weighted_temporal_mean(fire_wet).mean(axis=0))

# =============================================================================
# Fire map
# =============================================================================
min_lon = -70
max_lon =  -50 
min_lat =   -25  
max_lat = -5 

# set projections and initialise data
projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = fire_wet_mean.lon
latitude = fire_wet_mean.lat

data = [fire_wet_mean, fire_dry_mean]
labels = ['Wet season', 'Dry season']

# set display parameters
vmin = 0
vmax = 0.3
scaler = 1
cmap1 = plt.cm.get_cmap('Oranges')
levels = np.linspace(vmin*scaler, vmax*scaler, 21)
alpha = 1

# set up figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,8), subplot_kw={'projection':projection})
axes = axes.ravel()

# first panel (saved as im for colour bar creation)
im = axes[0].contourf(longitude, latitude, data[0]*scaler, levels = levels, cmap = cmap1, extend = 'max', alpha = alpha, transform=transform)

# repeat for other years
for i,y in enumerate(data):
    axes[i].set_title(labels[i], fontsize = 12)
    axes[i].coastlines()
    axes[i].add_feature(cfeature.BORDERS, edgecolor = 'gray')
    axes[i].contourf(longitude, latitude, data[i]*scaler, levels = levels, cmap = cmap1, extend = 'max', alpha = alpha, transform=transform)
    axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    gl = axes[i].gridlines(draw_labels=True)
    gl.xlocator = mticker.FixedLocator([-65, -60, -55])
    gl.ylocator = mticker.FixedLocator([-10, -20])

scalebar = ScaleBar(dx) #units = 'deg', dimension = 'angle')
plt.gca().add_artist(scalebar)
    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
# fig.subplots_adjust(top=0.9, right=0.8)

# add colorbar, vertical or horizontal
# cax = fig.add_axes([0.7, 0.05, 0.02, 0.85])
# cb = fig.colorbar(im0, cax=cax, orientation='vertical')
cax = fig.add_axes([0.1, 0.07, 0.8, 0.02])
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
# cb.set_label('Mean monthly burned area [10$^{2}$ % grid cell area]', fontsize = 12)

cb.set_label('GFED5 mean monthly burned area [% grid cell area]', fontsize = 12)

# save figure
# fig.savefig('.png', dpi = 300)

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
fire_wet_km = fire_wet_mean*crop_areas
fire_dry_km = fire_dry_mean*crop_areas


min_lon = -70
max_lon =  -50 
min_lat =   -25  
max_lat = -5 

# set projections and initialise data
projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = fire_wet_km.lon
latitude = fire_wet_km.lat

data = [fire_wet_km, fire_dry_km]
labels = ['Wet season', 'Dry season']

# set display parameters
vmin = 0
vmax = 1000
scaler = 1
cmap1 = plt.cm.get_cmap('Oranges')
levels = np.linspace(vmin*scaler, vmax*scaler, 25)
alpha = 1

# set up figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,8), subplot_kw={'projection':projection})
axes = axes.ravel()

# first panel (saved as im for colour bar creation)
im = axes[0].contourf(longitude, latitude, data[0]*scaler, levels = levels, cmap = cmap1, extend = 'max', alpha = alpha, transform=transform)

# repeat for other years
for i,y in enumerate(data):
    axes[i].set_title(labels[i], fontsize = 12)
    axes[i].coastlines()
    axes[i].add_feature(cfeature.BORDERS, edgecolor = 'gray')
    axes[i].contourf(longitude, latitude, data[i]*scaler, levels = levels, cmap = cmap1, extend = 'max', alpha = alpha, transform=transform)
    axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    gl = axes[i].gridlines(draw_labels=True)
    gl.xlocator = mticker.FixedLocator([-65, -60, -55])
    gl.ylocator = mticker.FixedLocator([-10, -20])

scalebar = ScaleBar(dx) #units = 'deg', dimension = 'angle')
plt.gca().add_artist(scalebar)
    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
# fig.subplots_adjust(top=0.9, right=0.8)

# add colorbar, vertical or horizontal
# cax = fig.add_axes([0.7, 0.05, 0.02, 0.85])
# cb = fig.colorbar(im0, cax=cax, orientation='vertical')
cax = fig.add_axes([0.1, 0.07, 0.8, 0.02])
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.set_label('Mean monthly burned area [km$^{2}$]', fontsize = 12)

# =============================================================================
# 4 panel plot: savanna+grassland, broadleaf, elevation, dry season burned area
# =============================================================================
min_lon = -70
max_lon =  -50 
min_lat =   -25  
max_lat = -5 

# set projections and initialise data
projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = broadleaf_mean.lon
latitude = broadleaf_mean.lat

data = [grass_sav, broadleaf_mean, elev, fire_dry_mean]
labels = ['(a) grasslands and savanna', '(b) broadleaf forest', '(c) elevation', '(d) burned area']

# set display parameters
vmin = 0
vmax = 1
scaler = 100
cmap1 = plt.cm.get_cmap('Greens')
levels = np.linspace(vmin*scaler, vmax*scaler, 11)

vmin2 = 0
vmax2 = 1000
scaler2 = 1
cmap2 = plt.cm.get_cmap('Blues')
levels2 = np.linspace(vmin2*scaler2, vmax2*scaler2, 11)

vmin3 = 0
vmax3 = 5000
scaler3 = 1
cmap3 = plt.cm.get_cmap('gist_earth')
levels3 = np.linspace(vmin3*scaler3, vmax3*scaler3, 21)

vmin4 = 0
vmax4 = 0.2
scaler4 = 1
cmap4 = plt.cm.get_cmap('Oranges')
levels4 = np.linspace(vmin4*scaler4, vmax4*scaler4, 21)

cm = 1/2.54
# set up figure
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12*cm,16*cm), subplot_kw={'projection':projection})
axes = axes.ravel()

# first panel (saved as im for colour bar creation)
im = axes[0].contourf(longitude, latitude, data[0]*scaler, levels = levels, cmap = cmap1, extend = 'neither', transform=transform)
im2 = axes[2].contourf(longitude, latitude, data[2]*scaler3, levels = levels3, cmap = cmap3, extend = 'both', transform=transform)
im3 = axes[3].contourf(longitude, latitude, data[3]*scaler4, levels = levels4, cmap = cmap4, extend = 'max', transform=transform)

# repeat for other years
for i,y in enumerate(data):
    axes[i].set_title(labels[i], fontsize = 10)
    axes[i].coastlines()
    if i == 2:
        axes[i].contourf(longitude, latitude, data[i]*scaler3, levels = levels3, cmap = cmap3, extend = 'both', transform=transform)
    elif i == 3:
        axes[i].contourf(longitude, latitude, data[i]*scaler4, levels = levels4, cmap = cmap4, extend = 'both', transform=transform)
    else:
        axes[i].contourf(longitude, latitude, data[i]*scaler, levels = levels, cmap = cmap1, extend = 'max', transform=transform)
        axes[i].contourf(longitude, latitude, high_elev, levels = levels2, cmap = cmap2, extend = 'max', alpha = 0.3, transform=transform )
    # axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    gl = axes[i].gridlines(draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-65, -55])
    gl.ylocator = mticker.FixedLocator([-10, -20])
    axes[i].add_feature(cfeature.BORDERS, zorder=10)

    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)

# add colorbar, vertical or horizontal
# cax = fig.add_axes([0.65, 0.05, 0.02, 0.87])
# cb = fig.colorbar(im, cax=cax, orientation='vertical')
cax = fig.add_axes([0.1, 0.24, 0.8, 0.01])
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.ax.tick_params(labelsize=8)
cax2 = fig.add_axes([0.1, 0.16, 0.8, 0.01])
cb2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
cb2.ax.tick_params(labelsize=8)
cax3 = fig.add_axes([0.1, 0.07, 0.8, 0.01])
cb3 = fig.colorbar(im3, cax=cax3, orientation='horizontal')
cb3.ax.tick_params(labelsize=8)


cb.set_label('Land type (% grid cell cover, subplots a, b)', fontsize = 8)
cb2.set_label('Elevation (m a.s.l., subplot c)', fontsize = 8)
# cb3.set_label('Dry season burned area (10$^{2}$ x % grid cell area, subplot d)', fontsize = 8)
cb3.set_label('Dry season burned area (% grid cell area, subplot d)', fontsize = 8)

# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/f02_4panel.png', dpi = 300)
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/f02_GFED5.png', dpi = 300)

