# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:27:58 2022

@author: s2261807
"""
###
# import functions
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

###
## load data

# fc
fn = 'R:\modis_lc\mcd12c1_1deg_2001-2019_forest_prop_corrected.nc'
ds = xr.open_dataset(fn, decode_times=True)
fc = ds['fc'][:]
ds.close()

abs_change = fc[-1,:,:] - fc[0,:,:]


# broadleaf forest cover
fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=True)
broadleaf = (ds['Land_Cover_Type_1_Percent'][:,:,:,2] + ds['Land_Cover_Type_1_Percent'][:,:,:,4] )/100
# broadleaf['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()
forest = broadleaf[-1,:,:]


### display
title = 'Forest cover change, 2019-2001'

longitude = fc.lon
latitude = fc.lat
data = abs_change[:,:]
# data = data.where(data != 0)

projection = ccrs.Robinson(central_longitude=0) #ccrs.PlateCarree() #ccrs.Robinson(central_longitude=0) #NorthPolarStereo(central_longitude=120) #ccrs.Robinson()
transform = ccrs.PlateCarree()

vmin = -0.5
vmax = 0.5
scaler = 100

cmap1 = plt.cm.get_cmap('seismic_r', 40)
newcolors = cmap1(np.linspace(0, 1, 40))
white = np.array([255/255, 255/255, 255/255, 1])
newcolors[19:21,:] = white
cmap2 = ListedColormap(newcolors)

levels = np.linspace(vmin*scaler, vmax*scaler, 41)


# initiate figure (with option for subplots)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), subplot_kw={'projection':projection})
#axes = axes.ravel()

# add coastlines, gridlines and title
axes.set_title(title)
axes.coastlines()
gl = axes.gridlines(draw_labels=True)
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])

# plot data
# im = axes.contourf(longitude, latitude, data*scaler, transform=transform, levels = levels, cmap = cmap1, extend = 'both') 
im = axes.contourf(longitude, latitude, data*scaler, transform=transform, cmap = cmap2, levels = levels, extend = 'both') 

# add, place, resize colorbar
cax = fig.add_axes([0.07, 0.09, 0.8, 0.02])
cb = plt.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label('Forest cover change [% grid cell area]')

# limit white space and ensure title visible
fig.tight_layout()
fig.subplots_adjust(top=0.92, right=0.9)

# save figure
# fig.savefig('M:\\figures\\land_cover\\fc_net_change_mcd12c1_global_2001-2018.png', dpi = 300)


# =============================================================================
# regional
# =============================================================================

### display
title = 'Forest cover change from 2001 to 2019'

longitude = fc.lon
latitude = fc.lat
data = abs_change[:,:]
# data = data.where(data != 0)

projection = ccrs.PlateCarree() #ccrs.PlateCarree() #ccrs.Robinson(central_longitude=0) #NorthPolarStereo(central_longitude=120) #ccrs.Robinson()
transform = ccrs.PlateCarree()

min_lon = -95 #60 #70 #-115 #-15 #-150 
max_lon =  -30 #180 # 150 #-30 #65 #-50 
min_lat =   -35 #30 #-15 #-35 #35 #25 
max_lat = 15 #90 #40 #25 #70 #65 

vmin = -0.5
vmax = 0.5
scaler = 100

cmap1 = plt.cm.get_cmap('seismic_r', 40)
newcolors = cmap1(np.linspace(0, 1, 40))
white = np.array([255/255, 255/255, 255/255, 1])
newcolors[19:21,:] = white
cmap2 = ListedColormap(newcolors)
levels = np.linspace(vmin*scaler, vmax*scaler, 41)

data2 = forest.where(forest > 0.5)
cmap3 = plt.cm.get_cmap('Greens', 2)
levels3 = np.linspace(50, 100, 2)



# initiate figure (with option for subplots)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), subplot_kw={'projection':projection})
#axes = axes.ravel()

# add coastlines, gridlines and title
axes.set_title(title, size=16)
axes.coastlines()
axes.add_feature(cfeature.BORDERS, edgecolor = 'gray')
gl = axes.gridlines(draw_labels=True, xlabel_style = {'size' : 14}, ylabel_style = {'size' : 14})
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = mticker.FixedLocator([-80, -60, -40])
gl.ylocator = mticker.FixedLocator(np.arange(-30, 11, 10))
axes.set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)

# plot data
# im = axes.contourf(longitude, latitude, data*scaler, transform=transform, levels = levels, cmap = cmap1, extend = 'both') 

im = axes.contourf(longitude, latitude, data*scaler, transform=transform, cmap = cmap2, levels = levels, extend = 'both') 

axes.contourf(longitude, latitude, forest*100, transform=transform, cmap = cmap3, levels = levels3, extend = 'neither', alpha = 0.2) 

## add region marker
axes.add_artist(Rectangle((-70, -25), 20, 20, fc="none", ec='indigo', ls = '--', lw = 3))

# add, place, resize colorbar
cax = fig.add_axes([0.82, 0.1, 0.02, 0.8])
cb = plt.colorbar(im, cax=cax, orientation="vertical")
cb.ax.tick_params(labelsize=12) 
cb.set_label('Forest cover change [% grid cell area]', size=16)

# limit white space and ensure title visible
fig.tight_layout()
fig.subplots_adjust(top=0.9, right=0.95)

# save figure
# fig.savefig('M:\\figures\\land_cover\\fc_net_change_mcd12c1_brazil_2001-2019_perccell_boarders_year2_poster_version.png', dpi = 300)
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/Region_may24.png', dpi = 300)

# =============================================================================
# EGU version
# =============================================================================

### display
longitude = fc.lon
latitude = fc.lat
data = abs_change[:,:]
# data = data.where(data != 0)

projection = ccrs.PlateCarree() #ccrs.PlateCarree() #ccrs.Robinson(central_longitude=0) #NorthPolarStereo(central_longitude=120) #ccrs.Robinson()
transform = ccrs.PlateCarree()

min_lon = -90 #60 #70 #-115 #-15 #-150 
max_lon =  -30 #180 # 150 #-30 #65 #-50 
min_lat =   -35 #30 #-15 #-35 #35 #25 
max_lat = 15 #90 #40 #25 #70 #65 

vmin = -0.5
vmax = 0.5
scaler = 100

cmap1 = plt.cm.get_cmap('seismic_r', 40)
newcolors = cmap1(np.linspace(0, 1, 40))
white = np.array([255/255, 255/255, 255/255, 1])
newcolors[19:21,:] = white
cmap2 = ListedColormap(newcolors)
levels = np.linspace(vmin*scaler, vmax*scaler, 41)

data2 = forest.where(forest > 0.5)
cmap3 = plt.cm.get_cmap('Greens', 2)
levels3 = np.linspace(50, 100, 2)


cm = 1/2.54
# initiate figure (with option for subplots)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 20*cm), subplot_kw={'projection':projection})
#axes = axes.ravel()

# add coastlines, gridlines and title
axes.coastlines()
axes.add_feature(cfeature.BORDERS, edgecolor = 'gray')
gl = axes.gridlines(draw_labels=True, xlabel_style = {'size' : 20}, ylabel_style = {'size' : 20})
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = mticker.FixedLocator([-80, -60, -40])
gl.ylocator = mticker.FixedLocator(np.arange(-30, 11, 10))
axes.set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)

# plot data
# im = axes.contourf(longitude, latitude, data*scaler, transform=transform, levels = levels, cmap = cmap1, extend = 'both') 

im = axes.contourf(longitude, latitude, data*scaler, transform=transform, cmap = cmap2, levels = levels, extend = 'both') 

axes.contourf(longitude, latitude, forest*100, transform=transform, cmap = cmap3, levels = levels3, extend = 'neither', alpha = 0.2) 

## add region marker
axes.add_artist(Rectangle((-70, -25), 20, 20, fc="none", ec='indigo', lw = 3))

# add, place, resize colorbar
cax = fig.add_axes([0.82, 0.1, 0.02, 0.8])
cb = plt.colorbar(im, cax=cax, orientation="vertical")
cb.ax.tick_params(labelsize=18) 
cb.set_label('Forest cover change [% grid cell area]', size=24)

# limit white space and ensure title visible
fig.tight_layout()
fig.subplots_adjust(right=0.86)

# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/Region_EGU.png', dpi = 300)

# =============================================================================
# 
# =============================================================================

# title = 'Forest cover change 2001 to 2018'

# longitude = fc.lon
# latitude = fc.lat
# data = abs_change[:,:]/fc[1,:,:]
# # data = data.where(data != 0)

# projection = ccrs.PlateCarree() #ccrs.PlateCarree() #ccrs.Robinson(central_longitude=0) #NorthPolarStereo(central_longitude=120) #ccrs.Robinson()
# transform = ccrs.PlateCarree()

# min_lon = -100 #60 #70 #-115 #-15 #-150 
# max_lon =  -20 #180 # 150 #-30 #65 #-50 
# min_lat =   -35 #30 #-15 #-35 #35 #25 
# max_lat = 15 #90 #40 #25 #70 #65 

# vmin = -1
# vmax = 1
# scaler = 100

# cmap1 = plt.cm.get_cmap('seismic_r', 40)
# newcolors = cmap1(np.linspace(0, 1, 40))
# white = np.array([255/255, 255/255, 255/255, 1])
# newcolors[19:21,:] = white
# cmap2 = ListedColormap(newcolors)

# levels = np.linspace(vmin*scaler, vmax*scaler, 41)


# # initiate figure (with option for subplots)
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 5), subplot_kw={'projection':projection})
# #axes = axes.ravel()

# # add coastlines, gridlines and title
# axes.set_title(title, size=14)
# axes.coastlines()
# gl = axes.gridlines(draw_labels=True)
# gl.xlocator = mticker.FixedLocator([-80, -60, -40])
# gl.ylocator = mticker.FixedLocator(np.arange(-30, 11, 10))
# axes.set_extent([int(min_lon), int(max_lon), int(min_lat), int(max_lat)], crs = transform)


# # plot data
# # im = axes.contourf(longitude, latitude, data*scaler, transform=transform, levels = levels, cmap = cmap1, extend = 'both') 
# im = axes.contourf(longitude, latitude, data*scaler, transform=transform, cmap = cmap2, levels = levels, extend = 'both') 

# # add, place, resize colorbar
# cax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
# cb = plt.colorbar(im, cax=cax, orientation="vertical")
# cb.set_label('Forest cover change [% 2001 forest area]', size=12)

# # limit white space and ensure title visible
# # fig.tight_layout()
# # fig.subplots_adjust(top=0.92, right=0.9)

# # save figure
# # fig.savefig('M:\\figures\\land_cover\\fc_net_change_mcd12c1_brazil_5yearmeans_perc2001.png', dpi = 300)
