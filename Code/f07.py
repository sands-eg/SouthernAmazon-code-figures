# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:44:58 2023

@author: s2261807


Script for summary figure 'heatmap style'

The data used has been previously processed and (in most cases) regridded to 1 by 1 degree resolution. 

Wet season = February through March
Dry season = August through October
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
from sklearn.linear_model import TheilSenRegressor
import seaborn as sns
# from datetime import datetime


### data

# burned area (GFED 4; 2001-2016); error in metadata - unit is fraction of cell, not %
fn = 'R:\\gfed\\monthly_1degree_sum_2001-2016.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire = ds['Burned area']*100 
fire['time'] = pd.date_range('2001-01-01', '2016-12-31', freq = 'MS')
ds.close()
# burned area GFED5
fn = 'R:\gfed\GFED5\GFED5_totalBA_2001-2020.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire2 = ds['__xarray_dataarray_variable__']#*100 
fire2['time'] = pd.date_range('2001-01-01', '2020-12-31', freq = 'MS')
ds.close()
## atmospheric composition

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

## DEM
fn = 'R:/DEM/SAM_DEM.nc'
ds = xr.open_dataset(fn)
dem = ds['SAM_DEM']
ds.close()  


### LC option B

fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=False)
grass_mosaic = ds['Land_Cover_Type_1_Percent'][:,:,:,10] + ds['Land_Cover_Type_1_Percent'][:,:,:,12] + ds['Land_Cover_Type_1_Percent'][:,:,:,14]
grass_mosaic['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()

grass = grass_mosaic

fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=True)
broadleaf = ds['Land_Cover_Type_1_Percent'][:,:,:,2] + ds['Land_Cover_Type_1_Percent'][:,:,:,4] 
broadleaf['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()

fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=False)
savanna = ds['Land_Cover_Type_1_Percent'][:,:,:,8:10].sum(axis=3)
savanna['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()
# =============================================================================
# define functions
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

def crop_ref_sector(var):
    ''' Function to crop data to area of interest:
        South America ()
        50-70 degrees W, 5-25 degrees S'''
    # mask_lon = (var.lon >= -83) & (var.lon <= -33)
    # mask_lat = (var.lat >= -56.5) & (var.lat <= 13.5)
    mask_lon = (var.lon >= -140) & (var.lon <= -100)
    mask_lat = (var.lat >= -30) & (var.lat <= 0)
    var_crop = var.where(mask_lon & mask_lat, drop=True)
    return var_crop

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

def mask_data(data, yes_fire, no_fire):
    data_fires = xr.zeros_like(data)
    data_no_fires = xr.zeros_like(data)
    
    if len(data.shape) == 2:
        data_fires[:,:] = data[:,:].where(yes_fire)
        data_no_fires[:,:] = data[:,:].where(no_fire)
    elif len(data.shape) == 3:
        for i in range(len(data[:,0,0])):
            data_fires[i,:,:] = data[i,:,:].where(yes_fire[i,:,:])
            data_no_fires[i,:,:] = data[i,:,:].where(no_fire[i,:,:])
    elif len(data.shape) == 4:
        for i in range(len(data[:,0,0,0])):
            for j in range(len(data[0,:,0,0])):
                data_fires[i,j,:,:] = data[i,j,:,:].where(yes_fire[i,:,:])
                data_no_fires[i,j,:,:] = data[i,j,:,:].where(no_fire[i,:,:])
    else: print('Invalid shape of data')
    return data_fires, data_no_fires

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

def calc_statistic(data, stat = 'median'):
    data_means = np.zeros(10)
    if stat == 'mean':
        for i,a in enumerate(data):
            data_means[i]=a.mean()
    elif stat == 'median':
        for i,a in enumerate(data):
            data_means[i]=np.median(a)  
    else:
        print('invalid stat')
    
    return data_means

def mask_high_elev(data, high_elev):
    
    data_low = xr.zeros_like(data)
    
    if len(data.shape) == 2:
        data_low[:,:] = data[:,:].where(~high_elev)
    elif len(data.shape) == 3:
        for i in range(len(data[:,0,0])):
            data_low[i,:,:] = data[i,:,:].where(~high_elev)
    else: print('Invalid shape of data')
    return data_low
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
# =============================================================================
# crop data to region of interest
# =============================================================================

fire_crop = crop_data(fire2)

hcho_crop = crop_data(hcho)
co_crop = crop_data(co)
no2_crop = crop_data(no2)
aod_crop = crop_data(aod)
isop_crop = crop_data(isop)
methanol_crop = crop_data(methanol)

broadleaf_crop = crop_data(broadleaf)
savanna_crop = crop_data(savanna)
grass_crop = crop_data(grass)

dem_crop = crop_data(dem)
# =============================================================================
# separate data into wet and dry season
# =============================================================================


# fire_wet = get_month_data([2,3,4], fire_crop)
fire_dry = get_month_data([8,9,10], fire_crop)

hcho_wet = get_month_data([2,3,4], hcho_crop)
hcho_dry = get_month_data([8,9,10], hcho_crop)

co_wet = get_month_data([2,3,4], co_crop)
co_dry = get_month_data([8,9,10], co_crop)

no2_wet = get_month_data([2,3,4], no2_crop)
no2_dry = get_month_data([8,9,10], no2_crop)

aod_wet = get_month_data([2,3,4], aod_crop)
aod_dry = get_month_data([8,9,10], aod_crop)

isop_wet = get_month_data([2,3,4], isop_crop)
isop_dry = get_month_data([8,9,10], isop_crop)

methanol_wet = get_month_data([2,3,4], methanol_crop)
methanol_dry = get_month_data([8,9,10], methanol_crop)
# =============================================================================
# detrend HCHO data
# =============================================================================

pacific = crop_ref_sector(hcho)
pac_weights = crop_ref_sector(spatial_weights)

pacific_wet = get_month_data([2,3,4], pacific)
pacific_dry = get_month_data([8,9,10], pacific)

wet_mean = weighted_temporal_mean(pacific_wet)
dry_mean = weighted_temporal_mean(pacific_dry)

pacific_mean_dry = spatial_weighted_average(dry_mean, pac_weights)
pacific_mean_wet = spatial_weighted_average(wet_mean, pac_weights)
# 

X = [i for i in range(0, len(pacific_mean_wet))]
X = np.reshape(X, (len(X), 1))
y1 = pacific_mean_wet
# Theil-Sen
model = TheilSenRegressor()
reg = model.fit(X, y1)
trend_TS_wet = model.predict(X)
R2_ts_wet = reg.score(X, y1)

X = [i for i in range(0, len(pacific_mean_dry))]
X = np.reshape(X, (len(X), 1))
y2 = pacific_mean_dry
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
# create high-fire and low-fire data subsets = mask my burned area - what value should be chosen? - start with any fire in a given month
# =============================================================================
atmos_no = 6
if atmos_no == 1:
    atmos_dry = isop_dry #aod_dry #no2_dry # co_dry #hcho_dry_d
    atmos_wet = isop_wet #aod_wet # no2_wet # co_wet #hcho_wet_d
elif atmos_no == 2:
    atmos_dry = aod_dry #no2_dry # co_dry #hcho_dry_d
    atmos_wet = aod_wet # no2_wet # co_wet #hcho_wet_d
elif atmos_no == 3:
    atmos_dry = methanol_dry
    atmos_wet = methanol_wet 
elif atmos_no == 4:
    atmos_dry = no2_dry # co_dry #hcho_dry_d
    atmos_wet = no2_wet # co_wet #hcho_wet_d
elif atmos_no == 5:
    atmos_dry = co_dry 
    atmos_wet = co_wet 
elif atmos_no == 6:
    atmos_dry = hcho_dry_d 
    atmos_wet = hcho_wet_d 
else:
    print('atmos_no is out of bounds (1 to 6)')

labels = {1 : 'Isoprene', 2 : 'AOD' , 3 : 'Methanol', 4 : 'NO$_{2}$', 5 : 'CO', 6 : 'HCHO'}

units = {1 : '[molecules cm$^{-2}$]', 2 : 'at 0.47 $\mu$m', 3 : '[ppbv]', 4 : '[molecules cm$^{-2}$]',\
         5 : '[10$^{17}$ molecules cm$^{-2}$]', 6 : '[molecules cm$^{-2}$]'}

lcs = [broadleaf_crop]
lc_labels = ['Broadleaf forest']

# lc_means = np.zeros((4, 10))

# elev_boundary = 1000
# high_elev = ma.masked_greater_equal(dem_crop, elev_boundary).mask

# atmos_dry = mask_high_elev(atmos_dry, high_elev) 
# atmos_wet = mask_high_elev(atmos_wet, high_elev)


plot_data = []

lcs = [broadleaf_crop, savanna_crop, grass_crop]
lc_labels = ['Broadleaf forest', 'Savanna', 'Grasses and croplands']
for j, y in enumerate(lcs):
    lc = y
    for i in range(1, 7):
        atmos_no = i
        if atmos_no == 1:
            atmos_dry = isop_dry #aod_dry #no2_dry # co_dry #hcho_dry_d
            atmos_wet = isop_wet #aod_wet # no2_wet # co_wet #hcho_wet_d
        elif atmos_no == 2:
            atmos_dry = methanol_dry
            atmos_wet = methanol_wet         
        elif atmos_no == 3:
            atmos_dry = hcho_dry_d 
            atmos_wet = hcho_wet_d 
        elif atmos_no == 4:
            atmos_dry = co_dry 
            atmos_wet = co_wet 
        elif atmos_no == 5:
            atmos_dry = aod_dry #no2_dry # co_dry #hcho_dry_d
            atmos_wet = aod_wet # no2_wet # co_wet #hcho_wet_d
        elif atmos_no == 6:
            atmos_dry = no2_dry # co_dry #hcho_dry_d
            atmos_wet = no2_wet # co_wet #hcho_wet_d
        else:
            print('atmos_no is out of bounds (1 to 6)')
            
        lc_means = np.zeros((2, 10))
    
        elev_boundary = 1000
        high_elev = ma.masked_greater_equal(dem_crop, elev_boundary).mask
    
        atmos_dry = mask_high_elev(atmos_dry, high_elev) 
        atmos_wet = mask_high_elev(atmos_wet, high_elev)
        
        ### keeping only data for months where both datasets are available
        min_time = xr.DataArray(data = [fire_dry.time.min().values, atmos_dry.time.min().values]).max().values
        min_year = min_time.astype(str).split('-')[0]
        max_time = xr.DataArray(data = [fire_dry.time.max().values, atmos_dry.time.max().values]).min().values
        max_year = max_time.astype(str).split('-')[0]
        
        fire_slice = fire_dry.sel(time=slice(min_year, max_year))
        atmos_dry_slice = atmos_dry.sel(time=slice(min_year, max_year))
        # atmos_wet_slice = atmos_wet.sel(time=slice(min_year, max_year))
        lc_slice = lc.sel(time=slice(min_year, max_year))
        #lc_int = (np.rint(lc_slice*100)).astype(int)
        lc_int = np.rint(lc_slice) #if LC B used
        
        ### masking fire occurrence
        boundary = 0.05#0.004
        no_fire = ma.masked_less_equal(fire_slice, boundary).mask
        yes_fire = ma.masked_greater(fire_slice, boundary).mask
            
        atmos_dry_fire, atmos_dry_no_fire = mask_data(atmos_dry_slice, yes_fire, no_fire)
        
        
        # =============================================================================
        # segregate data based on land cover percentage
        # =============================================================================
        reshape_lc = np.zeros_like(atmos_dry_slice)
        for a in range(lc_int.shape[0]):
            for b in range(3):
                reshape_lc[a*3+b,:,:] = lc_int[a]
                
        # confirm array shapes match
        if atmos_dry_fire.shape == reshape_lc.shape:
            print('Array shapes match')
        else:
            print('Array shapes differ - cannot use land cover as mask')
        
        # dry season + fire
        data1 = []
        data1_len = []
        
        for p in range(10, 101, 10): # or 51 for urban
            distribution_data = get_atm_dist(atmos_dry_fire, reshape_lc, p)
            distribution_data = distribution_data[~np.isnan(distribution_data)]
            data1.append(distribution_data)
            data1_len.append(len(distribution_data))
        
        # # dry season no fire
        # data2 = []
        # data2_len = []
        
        # for p in range(10, 101, 10): # or 51 for urban
        #     distribution_data = get_atm_dist(atmos_dry_no_fire, reshape_lc, p)
        #     distribution_data = distribution_data[~np.isnan(distribution_data)]
        #     data2.append(distribution_data)
        #     data2_len.append(len(distribution_data))
        
        # dry season all
        data3 = []
        data3_len = []
        
        for p in range(10, 101, 10): # or 51 for urban
            distribution_data = get_atm_dist(atmos_dry_slice, reshape_lc, p)
            distribution_data = distribution_data[~np.isnan(distribution_data)]
            data3.append(distribution_data)
            data3_len.append(len(distribution_data))
        
        # # wet season all
        # data4 = []
        # data4_len = []
        
        # for p in range(10, 101, 10): # or 51 for urban
        #     distribution_data = get_atm_dist(atmos_wet_slice, reshape_lc, p)
        #     distribution_data = distribution_data[~np.isnan(distribution_data)]
        #     data4.append(distribution_data)
        #     data4_len.append(len(distribution_data))
        
        
        # =============================================================================
        # calculate median (or other statistic) for each season/burned area/land cover bracket - add area weighting????
        # =============================================================================
            
        data1_means = calc_statistic(data1, stat='mean')
        # data2_means = calc_statistic(data2, stat='mean')
        data3_means = calc_statistic(data3, stat='mean')
        # data4_means = calc_statistic(data4, stat='mean')
        
        data1_cover = np.zeros(10)
        for l in range(10):
            data1_cover[l] = data1_len[l] /data3_len[l]
        
        data1_weighted = (data1_means * data1_len / data3_len ) / data3_means #- data1_cover
        # data2_weighted = data2_means * data2_len / data3_len
        
        
        plot_data.append(data1_weighted)

tmp_isop = np.stack((plot_data[0], plot_data[6], plot_data[12]))
tmp_met = np.stack((plot_data[1], plot_data[7], plot_data[13]))
tmp_hcho = np.stack((plot_data[2], plot_data[8], plot_data[14]))
tmp_co = np.stack((plot_data[3], plot_data[9], plot_data[15]))
tmp_aod = np.stack((plot_data[4], plot_data[10], plot_data[16]))
tmp_no2 = np.stack((plot_data[5], plot_data[11], plot_data[17]))

# # =============================================================================
# # heatmap plot
# # =============================================================================
# categories = ['High Fire', 'Low Fire', 'Dry', 'Wet']

# categories = ['H', 'L', 'D', 'W']

# cm = 1/2.54
# fig, axes = plt.subplots(figsize=(6 * cm, 3 * cm))
# axes.set_title(labels[atmos_no], fontsize = 8)
# axes = sns.heatmap(lc_means,  cmap='YlGnBu', cbar_kws = {'label':f'{labels[atmos_no]} {units[atmos_no]}'})#,
#                                                         # how to change fontsize' : '8' }) #vmin=10, vmax=30,

# # axes.set_ylabel('Category', fontsize = 10)
# axes.set_xlabel('Broadleaf forest % cover', fontsize = 8)
# axes.set_yticklabels(categories, fontsize=8)
# axes.set_xlim(0,10)
# axes.set_xticks(np.arange(2,11, 2))
# axes.set_xticklabels(range(20,101, 20), fontsize = 8)

# # fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/Summary/{labels[atmos_no]}_heatmap_notweighted_noneabove{elev_boundary}_broadleafB_hldw.png',\
# #             dpi = 300, bbox_inches='tight')

# =============================================================================
# broadleaf heatmaps for all 6 species
# =============================================================================
plot_data = [tmp_isop, tmp_met, tmp_hcho, tmp_co, tmp_aod, tmp_no2]

labels = {1 : 'Isoprene', 2 :  'Methanol', 3 : 'HCHO', 4 : 'CO', 5 : 'AOD', 6 : 'NO$_{2}$'}

units = {1 : '(10$^{16}$ molecules cm$^{-2}$)', 2 : '(ppbv)', 3 : '(10$^{16}$ molecules cm$^{-2}$)', 4 : '(10$^{17}$ molecules cm$^{-2}$)',\
          5 : 'at 0.47 $\mu$m', 6 : '(10$^{15}$ molecules cm$^{-2}$)'}


alphabet = ['a', 'b', 'c', 'd', 'e', 'f',]
categories = ['F', 'S', 'G']
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12*cm, 18*cm))
axes = axes.ravel()
cax1 = fig.add_axes([0.05, 0.10, 0.9, 0.02])

for i in range(6):
    axes[i].set_title(f'({alphabet[i]}) {labels[i+1]}', fontsize = 10)
    if i == 5:
    
        sns.heatmap(plot_data[i]*100,  cmap='seismic', vmin=0, vmax=100,  ax = axes[i],\
                    cbar_ax = cax1, cbar_kws = {'orientation' : 'horizontal'}) #'label':f'{labels[i+1]} {units[i+1]}', cbar_kws = {'location':'bottom'}, 
        cbar = axes[i].collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Proportion of regional mean from high fire grid cells (%)', fontsize = 8)
    else:
        sns.heatmap(plot_data[i]*100,  cmap='seismic', vmin=0, vmax=100,  ax = axes[i], cbar = False)
    #cbar_kws = {'label':f'{labels[i+1]} {units[i+1]}', 'location':'bottom'}
    # axes[i].text(-2, 0, f'({alphabet[i]})', fontsize = 10)
    axes[i].set_yticklabels(categories, fontsize=8)
    axes[i].set_xlim(0,10)
    axes[i].set_xticks(np.arange(2,11, 2))
    axes[i].set_xticklabels(range(20,101, 20), fontsize = 8)
    axes[i].set_xlabel('Land type cover (%)', fontsize = 8)
    # cbar = axes[i].collections[0].colorbar
    # cbar.ax.tick_params(labelsize=8)
    # cbar.set_label(f'{labels[i+1]} {units[i+1]}', fontsize = 8)

# add colorbars
# im1 = sns.heatmap(plot_data[0]*100,  cmap='seismic', vmin=0, vmax=100, cbar = True)
# cax1 = fig.add_axes([0.13, 0.20, 0.8, 0.01])
# cb1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
# cb1.ax.tick_params(labelsize=8)
# cb1.set_label('(a), (b): Isoprene (molecules cm$^{-2}$)', fontsize = 8)


fig.tight_layout()
fig.subplots_adjust(bottom=0.2)


# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/f07.png', dpi = 300)

# =============================================================================
# broadleaf heatmaps for all 6 species - resized
# =============================================================================
plot_data = [tmp_isop, tmp_met, tmp_hcho, tmp_co, tmp_aod, tmp_no2]

labels = {1 : 'Isoprene', 2 :  'Methanol', 3 : 'HCHO', 4 : 'CO', 5 : 'AOD', 6 : 'NO$_{2}$'}

units = {1 : '(10$^{16}$ molecules cm$^{-2}$)', 2 : '(ppbv)', 3 : '(10$^{16}$ molecules cm$^{-2}$)', 4 : '(10$^{17}$ molecules cm$^{-2}$)',\
          5 : 'at 0.47 $\mu$m', 6 : '(10$^{15}$ molecules cm$^{-2}$)'}


alphabet = ['a', 'b', 'c', 'd', 'e', 'f',]
categories = ['F', 'S', 'G']
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12*cm, 15*cm))
axes = axes.ravel()
cax1 = fig.add_axes([0.05, 0.10, 0.9, 0.02])

for i in range(6):
    axes[i].set_title(f'({alphabet[i]}) {labels[i+1]}', fontsize = 10)
    if i == 5:
    
        sns.heatmap(plot_data[i]*100,  cmap='seismic', vmin=0, vmax=100,  ax = axes[i],\
                    cbar_ax = cax1, cbar_kws = {'orientation' : 'horizontal'}) #'label':f'{labels[i+1]} {units[i+1]}', cbar_kws = {'location':'bottom'}, 
        cbar = axes[i].collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Proportion of regional mean from high fire grid cells (%)', fontsize = 8)
    else:
        sns.heatmap(plot_data[i]*100,  cmap='seismic', vmin=0, vmax=100,  ax = axes[i], cbar = False)
    #cbar_kws = {'label':f'{labels[i+1]} {units[i+1]}', 'location':'bottom'}
    # axes[i].text(-2, 0, f'({alphabet[i]})', fontsize = 10)
    axes[i].set_yticklabels(categories, fontsize=8)
    axes[i].set_xlim(0,10)
    axes[i].set_xticks(np.arange(2,11, 2))
    axes[i].set_xticklabels(range(20,101, 20), fontsize = 8)
    axes[i].set_xlabel('Land type cover (%)', fontsize = 8)
    # cbar = axes[i].collections[0].colorbar
    # cbar.ax.tick_params(labelsize=8)
    # cbar.set_label(f'{labels[i+1]} {units[i+1]}', fontsize = 8)

# add colorbars
# im1 = sns.heatmap(plot_data[0]*100,  cmap='seismic', vmin=0, vmax=100, cbar = True)
# cax1 = fig.add_axes([0.13, 0.20, 0.8, 0.01])
# cb1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
# cb1.ax.tick_params(labelsize=8)
# cb1.set_label('(a), (b): Isoprene (molecules cm$^{-2}$)', fontsize = 8)


fig.tight_layout()
fig.subplots_adjust(bottom=0.2)


# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/f07_resized_gfed5_b05.png', dpi = 300)
