# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:30:14 2023

@author: s2261807

Script for creating regional maps over the Southern Amazon:
    - mean percentage cover of savanna/grassland and forest, LAI and burned area

"""

# =============================================================================
# import functions and load data
# =============================================================================
#import packages and load data
import xarray as xr
import numpy as np
import pandas as pd

# plotting related:
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


# burned area (GFED 4; 2001-2016); error in metadata - unit is fraction of cell, not %
fn = 'R:\\gfed\\GFED4_BAkm2_2001-2016.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire = ds['Burned Area']
fire['time'] = pd.date_range('2001-01-01', '2016-12-31', freq = 'MS')
ds.close()


# grid cell area to convert burned area to %
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
fire = fire.fillna(0)/surface_area_earth *100 # fire converted to percentage

# LAI
fn = 'R:\modis_lai\modis_lai_2001-2019.nc'
ds = xr.open_dataset(fn, decode_times=True)
lai = ds['LAI'][:]/10
ds.close()

# land covers
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

# deforestation
fn = 'R:\modis_lc\mcd12c1_1deg_2001-2019_forest_prop_corrected.nc'
ds = xr.open_dataset(fn, decode_times=True)
fc = ds['fc'][:]
ds.close()
abs_change = fc[-1,:,:] - fc[0,:,:]

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
# calculate mean land surface fractions and dry season monthly fire
# =============================================================================

broadleaf_mean = crop_data(broadleaf).mean(axis=0)
grass_mean = crop_data(grass).mean(axis=0)
savanna_mean = crop_data(savanna).mean(axis=0)
elev = crop_data(dem)
high_elev = elev.where(elev >= 1000)
lai_mean = crop_data(lai).mean(axis=0)
grass_sav = grass_mean + savanna_mean

defo_crop = crop_data(abs_change)

fire_dry = get_month_data([8,9,10], fire)
fire_dry_mean = crop_data(weighted_temporal_mean(fire_dry).mean(axis=0))


# =============================================================================
# 4 panel plot: savanna+grassland, broadleaf, LAI, dry season burned area
# =============================================================================
### set projections and initialise data
projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = broadleaf_mean.lon
latitude = broadleaf_mean.lat

data = [grass_sav, broadleaf_mean, lai_mean, fire_dry_mean]
labels = ['(a) savanna and grassland', '(b) broadleaf forest', '(c) LAI', '(d) burned area']

### set display parameters
# preparing colormap for land cover and LAI
cmap1 = plt.cm.get_cmap('gist_earth_r')

# to specify divisions for land cover: extract all colors from the  map
cmaplist = [cmap1(i) for i in range(cmap1.N)]
# create the new map
cmap1a = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap1.N)
scaler = 100 # to display data as percentage
# define the bins and normalize
bounds = np.linspace(0, 100, 11)
norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)

# define colormap for high elevation polygon
vmin2 = 0.5
vmax2 = 1.5
scaler2 = 1
cmap2 = plt.cm.get_cmap('nipy_spectral')
levels2 = np.linspace(vmin2*scaler2, vmax2*scaler2, 11)

# colormap for burned area
cmap4 = plt.cm.get_cmap('magma_r')

# additional setting
fontsize = 8
cm = 1/2.54

### set up figure: number of subplots, figure size and geographic projection
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12*cm,16*cm), subplot_kw={'projection':projection})
axes = axes.ravel() # makes axis indexing easier

# panels (saved as im for colour bar creation)
im = axes[0].pcolormesh(longitude, latitude, data[0]*scaler,  vmin = 0, vmax = 100, cmap = cmap1a, norm = norm, transform=transform)#levels = levels,extend = 'neither', 
im1 = axes[1].pcolormesh(longitude, latitude, data[1]*scaler,  vmin = 0, vmax = 100, cmap = cmap1a, norm = norm, transform=transform)#levels = levels,extend = 'neither', 
im2 = axes[2].pcolormesh(longitude, latitude, data[2], vmin = 0, vmax = 5, cmap = cmap1, transform=transform)
im3 = axes[3].pcolormesh(longitude, latitude, data[3], vmin = 0, vmax = 0.3, cmap = cmap4, transform=transform)

# repeat for other years
for i,y in enumerate(data):
    axes[i].set_title(labels[i], fontsize = fontsize) # adding label/title to each subplot
    axes[i].coastlines() # adding coastlines
    if i == 3:# adding hatching over deforestation in burned area panel
        axes[i].contourf(longitude, latitude, -1*defo_crop, levels = np.linspace(0.025, 1, 2), colors = 'none', \
                        hatches=['//'], alpha = 0, transform=transform ) 
    # adding grey polygon over high elevation areas    
    axes[i].contourf(longitude, latitude, high_elev, levels = levels2, cmap = cmap2, extend = 'max', alpha = 1, transform=transform )
    # adding gridlines and county borders (with specifications)
    gl = axes[i].gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-65, -55])
    gl.ylocator = mticker.FixedLocator([-10, -20])
    gl.ylabel_style = {'fontsize' : fontsize}
    gl.xlabel_style = {'fontsize' : fontsize}
    axes[i].add_feature(cfeature.BORDERS, zorder=10)

    
# set title and layout
fig.tight_layout()
fig.subplots_adjust(bottom=0.25) # subplots adjusted to make space for colorbars

# add colorbars, vertical or horizontal
cax = fig.add_axes([0.1, 0.24, 0.8, 0.01])
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.ax.tick_params(labelsize=fontsize)
cax2 = fig.add_axes([0.1, 0.16, 0.8, 0.01])
cb2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
cb2.ax.tick_params(labelsize=fontsize)
cax3 = fig.add_axes([0.1, 0.07, 0.8, 0.01])
cb3 = fig.colorbar(im3, cax=cax3, orientation='horizontal')
cb3.ax.tick_params(labelsize=fontsize)

# add colorbar labels
cb.set_label('Land type (% grid cell cover, subplots a, b)', fontsize = fontsize)
cb2.set_label('LAI (subplot c)', fontsize = fontsize)
cb3.set_label('Dry season burned area (% grid cell area, subplot d)', fontsize = fontsize)

# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/surface_variable_maps_vjan2024.pdf') # can also be png etc.
