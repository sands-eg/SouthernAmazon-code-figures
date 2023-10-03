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
from sklearn.linear_model import TheilSenRegressor
from matplotlib_scalebar.scalebar import ScaleBar
from math import radians, sin, cos, acos

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

co_wet = get_month_data([2,3,4], co)
co_dry = get_month_data([8,9,10], co)
co_dry_mean = crop_data(weighted_temporal_mean(co_dry).mean(axis=0))
co_wet_mean = crop_data(weighted_temporal_mean(co_wet).mean(axis=0))

no2_wet = get_month_data([2,3,4], no2)
no2_dry = get_month_data([8,9,10], no2)
no2_dry_mean = crop_data(weighted_temporal_mean(no2_dry).mean(axis=0))
no2_wet_mean = crop_data(weighted_temporal_mean(no2_wet).mean(axis=0))

aod_wet = get_month_data([2,3,4], aod)
aod_dry = get_month_data([8,9,10], aod)
aod_dry_mean = crop_data(weighted_temporal_mean(aod_dry).mean(axis=0))
aod_wet_mean = crop_data(weighted_temporal_mean(aod_wet).mean(axis=0))

methanol_wet = get_month_data([2,3,4], methanol)
methanol_dry = get_month_data([8,9,10], methanol)
methanol_dry_mean = crop_data(weighted_temporal_mean(methanol_dry).mean(axis=0))
methanol_wet_mean = crop_data(weighted_temporal_mean(methanol_wet).mean(axis=0))
# methanol_dry_mean_global = weighted_temporal_mean(methanol_dry).mean(axis=0)
# methanol_wet_mean_global = weighted_temporal_mean(methanol_wet).mean(axis=0)

hcho_dry_mean = crop_data(weighted_temporal_mean(hcho_dry_d).mean(axis=0))
hcho_wet_mean = crop_data(weighted_temporal_mean(hcho_wet_d).mean(axis=0))

# # =============================================================================
# # display atmospheric composition maps
# # =============================================================================

# # set projections and initialise data
# projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
# transform = ccrs.PlateCarree()

# longitude = isop_dry_mean.lon
# latitude = isop_wet_mean.lat

# data = [isop_dry_mean, isop_wet_mean, hcho_dry_mean, hcho_wet_mean,\
#         co_dry_mean, co_wet_mean, aod_dry_mean, aod_wet_mean,\
#             no2_dry_mean, no2_wet_mean, methanol_dry_mean, methanol_wet_mean]
# labels = ['Isoprene', 'Isoprene', 'HCHO', 'HCHO', 'CO', 'CO', 'AOD', 'AOD', 'NO2', 'NO2', 'Methanol', 'Methanol']

# # set display parameters
# # vmin = 0
# # vmax = 100
# scaler = 1
# cmap1 = plt.cm.get_cmap('YlOrRd')
# # levels = np.linspace(vmin*scaler, vmax*scaler, 11)

# # set up figure
# fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12,12), subplot_kw={'projection':projection})
# axes = axes.ravel()

# # first panel (saved as im for colour bar creation)
# im = axes[0].contourf(longitude, latitude, data[0]*scaler, cmap = cmap1, extend = 'neither', transform=transform) #levels = levels, 

# # repeat for other years
# for i,y in enumerate(data):
#     axes[i].set_title(labels[i], fontsize = 12)
#     axes[i].coastlines()
#     axes[i].contourf(longitude, latitude, data[i]*scaler, cmap = cmap1, extend = 'neither', transform=transform) #levels = levels, 
# #     axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
# #     gl = axes[i].gridlines(draw_labels=True)
# #     gl.xlocator = mticker.FixedLocator([-65, -60, -55])
# #     gl.ylocator = mticker.FixedLocator([-10, -20])

# # scalebar = ScaleBar(dx) #units = 'deg', dimension = 'angle')
# # plt.gca().add_artist(scalebar)
    
# # set title and layout
# # fig.suptitle('Mean cover %', fontsize = 16)
# fig.tight_layout()
# # fig.subplots_adjust(top=0.92, right=0.9)

# # add colorbar, vertical or horizontal
# # cax = fig.add_axes([0.9, 0.05, 0.02, 0.87])
# # cb = fig.colorbar(im, cax=cax, orientation='vertical')
# # # cax = fig.add_axes([0.1, 0, 0.8, 0.02])
# # # cb = fig.colorbar(im, cax=cax, orientation='horizontal')

# cb.set_label('% grid cell cover', fontsize = 12)

# # save figure
# # fig.savefig('.png', dpi = 300)

# =============================================================================
# compare wet and dry season
# =============================================================================
atmos_no = 4
if atmos_no == 1:
    atmos_dry = isop_dry_mean #aod_dry #no2_dry # co_dry #hcho_dry_d
    atmos_wet = isop_wet_mean #aod_wet # no2_wet # co_wet #hcho_wet_d
elif atmos_no == 2:
    atmos_dry = aod_dry_mean #no2_dry # co_dry #hcho_dry_d
    atmos_wet = aod_wet_mean # no2_wet # co_wet #hcho_wet_d
elif atmos_no == 3:
    atmos_dry = methanol_dry_mean
    atmos_wet = methanol_wet_mean
elif atmos_no == 4:
    atmos_dry = no2_dry_mean # co_dry #hcho_dry_d
    atmos_wet = no2_wet_mean # co_wet #hcho_wet_d
elif atmos_no == 5:
    atmos_dry = co_dry_mean
    atmos_wet = co_wet_mean 
elif atmos_no == 6:
    atmos_dry = hcho_dry_mean 
    atmos_wet = hcho_wet_mean 
else:
    print('atmos_no is out of bounds (1 to 6)')

labels = {1 : 'Isoprene', 2 : 'AOD' , 3 : 'Methanol', 4 : 'NO$_{2}$', 5 : 'CO', 6 : 'HCHO'}
units = {1 : '[molecules cm$^{-2}$]', 2 : 'at 0.47 $\mu$m', 3 : '[ppbv]', 4 : '[molecules cm$^{-2}$]',\
         5 : '[10$^{17}$ molecules cm$^{-2}$]', 6 : '[molecules cm$^{-2}$]'}


# projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
# transform = ccrs.PlateCarree()

# longitude = isop_dry_mean.lon
# latitude = isop_wet_mean.lat

# data = [atmos_wet, atmos_dry]
# #, hcho_dry_mean, hcho_wet_mean,\
# #        co_dry_mean, co_wet_mean, aod_dry_mean, aod_wet_mean,\
# #            no2_dry_mean, no2_wet_mean, methanol_dry_mean, methanol_wet_mean]
# subplot_labels = ['Wet season', 'Dry season'] #, 'HCHO', 'HCHO', 'CO', 'CO', 'AOD', 'AOD', 'NO2', 'NO2', 'Methanol', 'Methanol']

# # set display parameters
# vmin = 0
# vmax = 3
# scaler = 10**15
# cmap1 = plt.cm.get_cmap('YlOrRd')
# levels = np.linspace(vmin*scaler, vmax*scaler, 11)

# vmin2 = 0
# vmax2 = 1000
# scaler2 = 1
# cmap2 = plt.cm.get_cmap('Blues')
# levels2 = np.linspace(vmin2*scaler2, vmax2*scaler2, 11)

# # set up figure
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7), subplot_kw={'projection':projection})
# axes = axes.ravel()

# # first panel (saved as im for colour bar creation)
# im = axes[0].contourf(longitude, latitude, data[0], levels = levels, cmap = cmap1, extend = 'max', transform=transform) #

# # repeat for other years
# for i,y in enumerate(data):
#     axes[i].set_title(subplot_labels[i], fontsize = 12)
#     axes[i].coastlines()
#     axes[i].contourf(longitude, latitude, data[i], levels = levels, cmap = cmap1, extend = 'max', transform=transform) #levels = levels, 
#     axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
#     axes[i].contourf(longitude, latitude, high_elev, levels = levels2, cmap = cmap2, extend = 'max', alpha = 0.3, transform=transform )
#     gl = axes[i].gridlines(draw_labels=True)
#     gl.xlocator = mticker.FixedLocator([-65, -60, -55])
#     gl.ylocator = mticker.FixedLocator([-10, -20])

# scalebar = ScaleBar(dx) #units = 'deg', dimension = 'angle')
# plt.gca().add_artist(scalebar)
    
# # set title and layout
# # fig.suptitle('Mean cover %', fontsize = 16)
# fig.tight_layout()
# fig.subplots_adjust(bottom=0.1)

# # add colorbar, vertical or horizontal
# # cax = fig.add_axes([0.9, 0.05, 0.02, 0.87])
# # cb = fig.colorbar(im, cax=cax, orientation='vertical')
# cax = fig.add_axes([0.1, 0.1, 0.8, 0.02])
# cb = fig.colorbar(im, cax=cax, orientation='horizontal')

# cb.set_label(f'{labels[atmos_no]} {units[atmos_no]}', fontsize = 12)

# # save figure
# # fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/Summary/{labels[atmos_no]}_maps_whighelev.png', dpi = 300)


# =============================================================================
# f03.png Isoprene, Methanol and Formaldehyde seasonal maps
# =============================================================================
min_lon = -70
max_lon =  -50 
min_lat =   -25  
max_lat = -5 

projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = isop_dry_mean.lon
latitude = isop_wet_mean.lat

data = [isop_wet_mean, isop_dry_mean, methanol_wet_mean, methanol_dry_mean, hcho_wet_mean, hcho_dry_mean]
#, hcho_dry_mean, hcho_wet_mean,\
#        co_dry_mean, co_wet_mean, aod_dry_mean, aod_wet_mean,\
#            no2_dry_mean, no2_wet_mean, methanol_dry_mean, methanol_wet_mean]
subplot_labels = ['Wet season', 'Dry season'] #, 'HCHO', 'HCHO', 'CO', 'CO', 'AOD', 'AOD', 'NO2', 'NO2', 'Methanol', 'Methanol']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f',]
# set display parameters
vmin1 = 0
vmax1 = 2
scaler1 = 10**16
cmap1 = plt.cm.get_cmap('YlOrRd')
levels1 = np.linspace(vmin1*scaler1, vmax1*scaler1, 11)

vmin3 = 0
vmax3 = 1
scaler3 = 1
cmap3 = plt.cm.get_cmap('YlOrRd')
levels3 = np.linspace(vmin3*scaler3, vmax3*scaler3, 11)

vmin4 = 0
vmax4 = 2
scaler4 = 10**16
cmap4 = plt.cm.get_cmap('YlOrRd')
levels4 = np.linspace(vmin4*scaler4, vmax4*scaler4, 11)


vmin2 = 0
vmax2 = 1000
scaler2 = 1
cmap2 = plt.cm.get_cmap('Greys')
levels2 = np.linspace(vmin2*scaler2, vmax2*scaler2, 11)

# set up figure
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12*cm, 20*cm), subplot_kw={'projection':projection})
axes = axes.ravel()

# first panel (saved as im for colour bar creation)
im1 = axes[0].contourf(longitude, latitude, data[0], levels = levels1, cmap = cmap1, extend = 'max', transform=transform)
im2 = axes[2].contourf(longitude, latitude, data[2], levels = levels3, cmap = cmap3, extend = 'max', transform=transform)
im3 = axes[4].contourf(longitude, latitude, data[4], levels = levels4, cmap = cmap4, extend = 'max', transform=transform) #

# repeat for other years
for i,y in enumerate(data):
    axes[i].coastlines()
    if i < 2:
        axes[i].set_title(subplot_labels[i], fontsize = 10)
        axes[i].contourf(longitude, latitude, data[i], levels = levels1, cmap = cmap1, extend = 'max', transform=transform) #levels = levels, 
    elif i < 4 and i > 1:
        axes[i].contourf(longitude, latitude, data[i], levels = levels3, cmap = cmap3, extend = 'max', transform=transform) 
    elif i > 3:
        axes[i].contourf(longitude, latitude, data[i], levels = levels4, cmap = cmap4, extend = 'max', transform=transform)
    axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    axes[i].contourf(longitude, latitude, high_elev, levels = levels2, cmap = cmap2, extend = 'max', alpha = 0.7, transform=transform )
    gl = axes[i].gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-65, -55])
    gl.ylocator = mticker.FixedLocator([-10, -20])
    gl.xlabel_style = {'size' : 8}
    gl.ylabel_style = {'size' : 8}
    axes[i].add_feature(cfeature.BORDERS, zorder=10)
    axes[i].text(-73, -7, f'({alphabet[i]})', fontsize = 10)


scalebar = ScaleBar(dx, box_alpha=0.6)#, location = 'lower right', box_color='white') #units = 'deg', dimension = 'angle')
plt.gca().add_artist(scalebar)
    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)

# add colorbars
cax1 = fig.add_axes([0.13, 0.20, 0.8, 0.01])
cb1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
cb1.ax.tick_params(labelsize=8)
cb1.set_label('(a), (b): Isoprene (molecules cm$^{-2}$)', fontsize = 8)

cax2 = fig.add_axes([0.13, 0.13, 0.8, 0.01])
cb2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
cb2.ax.tick_params(labelsize=8)
cb2.set_label('(c), (d): Methanol (ppbv)', fontsize = 8)

cax3 = fig.add_axes([0.13, 0.06, 0.8, 0.01])
cb3 = fig.colorbar(im3, cax=cax3, orientation='horizontal')
cb3.ax.tick_params(labelsize=8)
cb3.set_label('(e), (f): Formaldehyde (molecules cm$^{-2}$)', fontsize = 8)

# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/f03.png', dpi = 300)


# =============================================================================
# f04.png Carbon monoxide, AOD, NO2 seasonal maps
# =============================================================================
min_lon = -70
max_lon =  -50 
min_lat =   -25  
max_lat = -5 

projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = co_dry_mean.lon
latitude = co_wet_mean.lat

data = [co_wet_mean, co_dry_mean, aod_wet_mean, aod_dry_mean, no2_wet_mean, no2_dry_mean]
#, hcho_dry_mean, hcho_wet_mean,\
#        co_dry_mean, co_wet_mean, aod_dry_mean, aod_wet_mean,\
#            no2_dry_mean, no2_wet_mean, methanol_dry_mean, methanol_wet_mean]
subplot_labels = ['Wet season', 'Dry season'] #, 'HCHO', 'HCHO', 'CO', 'CO', 'AOD', 'AOD', 'NO2', 'NO2', 'Methanol', 'Methanol']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f',]
# set display parameters
vmin1 = 0
vmax1 = 4
scaler1 = 10
cmap1 = plt.cm.get_cmap('YlOrRd')
levels1 = np.linspace(vmin1*scaler1, vmax1*scaler1, 11)

vmin3 = 0
vmax3 = 1
scaler3 = 1
cmap3 = plt.cm.get_cmap('YlOrRd')
levels3 = np.linspace(vmin3*scaler3, vmax3*scaler3, 11)

vmin4 = 0
vmax4 = 3
scaler4 = 10**15
cmap4 = plt.cm.get_cmap('YlOrRd')
levels4 = np.linspace(vmin4*scaler4, vmax4*scaler4, 11)


vmin2 = 0
vmax2 = 1000
scaler2 = 1
cmap2 = plt.cm.get_cmap('Greys')
levels2 = np.linspace(vmin2*scaler2, vmax2*scaler2, 11)

# set up figure
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12*cm, 20*cm), subplot_kw={'projection':projection})
axes = axes.ravel()

# first panel (saved as im for colour bar creation)
im1 = axes[0].contourf(longitude, latitude, data[0], levels = levels1, cmap = cmap1, extend = 'max', transform=transform)
im2 = axes[2].contourf(longitude, latitude, data[2], levels = levels3, cmap = cmap3, extend = 'max', transform=transform)
im3 = axes[4].contourf(longitude, latitude, data[4], levels = levels4, cmap = cmap4, extend = 'max', transform=transform) #

# repeat for other years
for i,y in enumerate(data):
    axes[i].coastlines()
    if i < 2:
        axes[i].set_title(subplot_labels[i], fontsize = 10)
        axes[i].contourf(longitude, latitude, data[i], levels = levels1, cmap = cmap1, extend = 'max', transform=transform) #levels = levels, 
    elif i < 4 and i > 1:
        axes[i].contourf(longitude, latitude, data[i], levels = levels3, cmap = cmap3, extend = 'max', transform=transform) 
    elif i > 3:
        axes[i].contourf(longitude, latitude, data[i], levels = levels4, cmap = cmap4, extend = 'max', transform=transform)
    axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    axes[i].contourf(longitude, latitude, high_elev, levels = levels2, cmap = cmap2, extend = 'max', alpha = 0.7, transform=transform )
    gl = axes[i].gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-65, -55])
    gl.ylocator = mticker.FixedLocator([-10, -20])
    gl.xlabel_style = {'size' : 8}
    gl.ylabel_style = {'size' : 8}
    axes[i].add_feature(cfeature.BORDERS, zorder=10)
    axes[i].text(-73, -7, f'({alphabet[i]})', fontsize = 10)


scalebar = ScaleBar(dx, box_alpha=0.6)#, location = 'lower right', box_color='white') #units = 'deg', dimension = 'angle')
plt.gca().add_artist(scalebar)
    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)

# add colorbars
cax1 = fig.add_axes([0.13, 0.20, 0.8, 0.01])
cb1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
cb1.ax.tick_params(labelsize=8)
cb1.set_label('(a), (b): Carbon monoxide (10$^{17}$ molecules cm$^{-2}$)', fontsize = 8)

cax2 = fig.add_axes([0.13, 0.13, 0.8, 0.01])
cb2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
cb2.ax.tick_params(labelsize=8)
cb2.set_label('(c), (d): Aerosol optical depth at 0.47 $\mu$m', fontsize = 8)

cax3 = fig.add_axes([0.13, 0.06, 0.8, 0.01])
cb3 = fig.colorbar(im3, cax=cax3, orientation='horizontal')
cb3.ax.tick_params(labelsize=8)
cb3.set_label('(e), (f): Nitrogen dioxide (molecules cm$^{-2}$)', fontsize = 8)

# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/f04.png', dpi = 300)


# =============================================================================
# f03.png Isoprene, Methanol and Formaldehyde seasonal maps - resized 
# =============================================================================
min_lon = -70
max_lon =  -50 
min_lat =   -25  
max_lat = -5 

projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = isop_dry_mean.lon
latitude = isop_wet_mean.lat

data = [isop_wet_mean, isop_dry_mean, methanol_wet_mean, methanol_dry_mean, hcho_wet_mean, hcho_dry_mean]
#, hcho_dry_mean, hcho_wet_mean,\
#        co_dry_mean, co_wet_mean, aod_dry_mean, aod_wet_mean,\
#            no2_dry_mean, no2_wet_mean, methanol_dry_mean, methanol_wet_mean]
subplot_labels = ['Wet season', 'Dry season'] #, 'HCHO', 'HCHO', 'CO', 'CO', 'AOD', 'AOD', 'NO2', 'NO2', 'Methanol', 'Methanol']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f',]
# set display parameters
vmin1 = 0
vmax1 = 2
scaler1 = 10**16
cmap1 = plt.cm.get_cmap('YlOrRd')
levels1 = np.linspace(vmin1*scaler1, vmax1*scaler1, 11)

vmin3 = 0
vmax3 = 1
scaler3 = 1
cmap3 = plt.cm.get_cmap('YlOrRd')
levels3 = np.linspace(vmin3*scaler3, vmax3*scaler3, 11)

vmin4 = 0
vmax4 = 2
scaler4 = 10**16
cmap4 = plt.cm.get_cmap('YlOrRd')
levels4 = np.linspace(vmin4*scaler4, vmax4*scaler4, 11)


vmin2 = 0
vmax2 = 1000
scaler2 = 1
cmap2 = plt.cm.get_cmap('Greys')
levels2 = np.linspace(vmin2*scaler2, vmax2*scaler2, 11)

# set up figure
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12*cm, 14*cm), subplot_kw={'projection':projection})
axes = axes.ravel()

# first panel (saved as im for colour bar creation)
im1 = axes[0].contourf(longitude, latitude, data[0]/10**16, levels = levels1/10**16, cmap = cmap1, extend = 'max', transform=transform)
im2 = axes[2].contourf(longitude, latitude, data[2], levels = levels3, cmap = cmap3, extend = 'max', transform=transform)
im3 = axes[4].contourf(longitude, latitude, data[4]/10**16, levels = levels4/10**16, cmap = cmap4, extend = 'max', transform=transform) #

# repeat for other years
for i,y in enumerate(data):
    axes[i].coastlines()
    if i < 2:
        axes[i].set_title(subplot_labels[i], fontsize = 10)
        axes[i].contourf(longitude, latitude, data[i], levels = levels1, cmap = cmap1, extend = 'max', transform=transform) #levels = levels, 
    elif i < 4 and i > 1:
        axes[i].contourf(longitude, latitude, data[i], levels = levels3, cmap = cmap3, extend = 'max', transform=transform) 
    elif i > 3:
        axes[i].contourf(longitude, latitude, data[i], levels = levels4, cmap = cmap4, extend = 'max', transform=transform)
    axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    axes[i].contourf(longitude, latitude, high_elev, levels = levels2, cmap = cmap2, extend = 'max', alpha = 0.7, transform=transform )
    gl = axes[i].gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-65, -55])
    gl.ylocator = mticker.FixedLocator([-10, -20])
    gl.xlabel_style = {'size' : 8}
    gl.ylabel_style = {'size' : 8}
    axes[i].add_feature(cfeature.BORDERS, zorder=10)
    axes[i].text(-73, -7, f'({alphabet[i]})', fontsize = 10)


scalebar = ScaleBar(dx, box_alpha=0.6)#, location = 'lower right', box_color='white') #units = 'deg', dimension = 'angle')
plt.gca().add_artist(scalebar)
    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
fig.subplots_adjust(right=0.8)

# add colorbars
cax1 = fig.add_axes([0.8, 0.68, 0.02, 0.25])
cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
cb1.ax.tick_params(labelsize=8)
cb1.set_label('(a), (b): Isoprene \n(10$^{16}$ molecules cm$^{-2}$)', fontsize = 8)

cax2 = fig.add_axes([0.8, 0.37, 0.02, 0.25])
cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
cb2.ax.tick_params(labelsize=8)
cb2.set_label('(c), (d): Methanol (ppbv)', fontsize = 8)

cax3 = fig.add_axes([0.8, 0.05, 0.02, 0.25])
cb3 = fig.colorbar(im3, cax=cax3, orientation='vertical')
cb3.ax.tick_params(labelsize=8)
cb3.set_label('(e), (f): Formaldehyde \n(10$^{16}$ molecules cm$^{-2}$)', fontsize = 8)

# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/f03_resized.png', dpi = 300)


# =============================================================================
# f04.png Carbon monoxide, AOD, NO2 seasonal maps - resized
# =============================================================================
min_lon = -70
max_lon =  -50 
min_lat =   -25  
max_lat = -5 

projection = ccrs.PlateCarree() #ccrs.Robinson() # or ccrs.PlateCarree() for faster drawing?
transform = ccrs.PlateCarree()

longitude = co_dry_mean.lon
latitude = co_wet_mean.lat

data = [co_wet_mean, co_dry_mean, aod_wet_mean, aod_dry_mean, no2_wet_mean, no2_dry_mean]
#, hcho_dry_mean, hcho_wet_mean,\
#        co_dry_mean, co_wet_mean, aod_dry_mean, aod_wet_mean,\
#            no2_dry_mean, no2_wet_mean, methanol_dry_mean, methanol_wet_mean]
subplot_labels = ['Wet season', 'Dry season'] #, 'HCHO', 'HCHO', 'CO', 'CO', 'AOD', 'AOD', 'NO2', 'NO2', 'Methanol', 'Methanol']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f',]
# set display parameters
vmin1 = 0
vmax1 = 4
scaler1 = 10
cmap1 = plt.cm.get_cmap('YlOrRd')
levels1 = np.linspace(vmin1*scaler1, vmax1*scaler1, 11)

vmin3 = 0
vmax3 = 1
scaler3 = 1
cmap3 = plt.cm.get_cmap('YlOrRd')
levels3 = np.linspace(vmin3*scaler3, vmax3*scaler3, 11)

vmin4 = 0
vmax4 = 3
scaler4 = 10**15
cmap4 = plt.cm.get_cmap('YlOrRd')
levels4 = np.linspace(vmin4*scaler4, vmax4*scaler4, 11)


vmin2 = 0
vmax2 = 1000
scaler2 = 1
cmap2 = plt.cm.get_cmap('Greys')
levels2 = np.linspace(vmin2*scaler2, vmax2*scaler2, 11)

# set up figure
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12*cm, 14*cm), subplot_kw={'projection':projection})
axes = axes.ravel()

# first panel (saved as im for colour bar creation)
im1 = axes[0].contourf(longitude, latitude, data[0], levels = levels1, cmap = cmap1, extend = 'max', transform=transform)
im2 = axes[2].contourf(longitude, latitude, data[2], levels = levels3, cmap = cmap3, extend = 'max', transform=transform)
im3 = axes[4].contourf(longitude, latitude, data[4]/10**15, levels = levels4/10**15, cmap = cmap4, extend = 'max', transform=transform) #

# repeat for other years
for i,y in enumerate(data):
    axes[i].coastlines()
    if i < 2:
        axes[i].set_title(subplot_labels[i], fontsize = 10)
        axes[i].contourf(longitude, latitude, data[i], levels = levels1, cmap = cmap1, extend = 'max', transform=transform) #levels = levels, 
    elif i < 4 and i > 1:
        axes[i].contourf(longitude, latitude, data[i], levels = levels3, cmap = cmap3, extend = 'max', transform=transform) 
    elif i > 3:
        axes[i].contourf(longitude, latitude, data[i], levels = levels4, cmap = cmap4, extend = 'max', transform=transform)
    axes[i].set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)
    axes[i].contourf(longitude, latitude, high_elev, levels = levels2, cmap = cmap2, extend = 'max', alpha = 0.7, transform=transform )
    gl = axes[i].gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-65, -55])
    gl.ylocator = mticker.FixedLocator([-10, -20])
    gl.xlabel_style = {'size' : 8}
    gl.ylabel_style = {'size' : 8}
    axes[i].add_feature(cfeature.BORDERS, zorder=10)
    axes[i].text(-73, -7, f'({alphabet[i]})', fontsize = 10)


scalebar = ScaleBar(dx, box_alpha=0.6)#, location = 'lower right', box_color='white') #units = 'deg', dimension = 'angle')
plt.gca().add_artist(scalebar)
    
# set title and layout
# fig.suptitle('Mean cover %', fontsize = 16)
fig.tight_layout()
fig.subplots_adjust(right=0.8)

# add colorbars
cax1 = fig.add_axes([0.8, 0.68, 0.02, 0.25])
cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
cb1.ax.tick_params(labelsize=8)
cb1.set_label('(a), (b): Carbon monoxide \n(10$^{17}$ molecules cm$^{-2}$)', fontsize = 8)

cax2 = fig.add_axes([0.8, 0.37, 0.02, 0.25])
cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
cb2.ax.tick_params(labelsize=8)
cb2.set_label('(c), (d): Aerosol optical depth \nat 0.47 $\mu$m', fontsize = 8)

cax3 = fig.add_axes([0.8, 0.05, 0.02, 0.25])
cb3 = fig.colorbar(im3, cax=cax3, orientation='vertical')
cb3.ax.tick_params(labelsize=8)
cb3.set_label('(e), (f): Nitrogen dioxide \n(10$^{15}$ molecules cm$^{-2}$)', fontsize = 8)

# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/f04_resized.png', dpi = 300)
