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
ds = xr.open_dataset(fn, decode_times=True)
broadleaf = ds['Land_Cover_Type_1_Percent'][:,:,:,2] + ds['Land_Cover_Type_1_Percent'][:,:,:,4] 
broadleaf['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()

### fire
fn = 'R:\\gfed\\GFED4_BAkm2_2001-2016.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire = ds['Burned Area']
fire['time'] = pd.date_range('2001-01-01', '2016-12-31', freq = 'MS')
ds.close()


# # burned area GFED5
# fn = 'R:\gfed\GFED5\GFED5_totalBA_2001-2020.nc'
# ds = xr.open_dataset(fn, decode_times=True)
# fire2 = ds['__xarray_dataarray_variable__'].fillna(0)#*100 
# fire2['time'] = pd.date_range('2001-01-01', '2020-12-31', freq = 'MS')
# ds.close()

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
    elif stat == 'std':
        for i,a in enumerate(data):
            data_means[i] = a.std()
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

# fire2 = fire2.fillna(0)/surface_area_earth *100
fire = fire.fillna(0)/surface_area_earth *100
# =============================================================================
# crop data to region of interest
# =============================================================================
fire_crop = crop_data(fire)
# fire2_crop = crop_data(fire2)

hcho_crop = crop_data(hcho)
co_crop = crop_data(co)
no2_crop = crop_data(no2)
aod_crop = crop_data(aod)
isop_crop = crop_data(isop)
methanol_crop = crop_data(methanol)

broadleaf_crop = crop_data(broadleaf)

dem_crop = crop_data(dem)
# =============================================================================
# separate data into wet and dry season
# =============================================================================

fire_wet = get_month_data([2,3,4], fire_crop)
# fire2_wet = get_month_data([2,3,4], fire2_crop)
fire_dry = get_month_data([8,9,10], fire_crop)
# fire2_dry = get_month_data([8,9,10], fire2_crop)

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
# atmos_no = 6
# if atmos_no == 1:
#     atmos_dry = isop_dry #aod_dry #no2_dry # co_dry #hcho_dry_d
#     atmos_wet = isop_wet #aod_wet # no2_wet # co_wet #hcho_wet_d
# elif atmos_no == 2:
#     atmos_dry = aod_dry #no2_dry # co_dry #hcho_dry_d
#     atmos_wet = aod_wet # no2_wet # co_wet #hcho_wet_d
# elif atmos_no == 3:
#     atmos_dry = methanol_dry
#     atmos_wet = methanol_wet 
# elif atmos_no == 4:
#     atmos_dry = no2_dry # co_dry #hcho_dry_d
#     atmos_wet = no2_wet # co_wet #hcho_wet_d
# elif atmos_no == 5:
#     atmos_dry = co_dry 
#     atmos_wet = co_wet 
# elif atmos_no == 6:
#     atmos_dry = hcho_dry_d 
#     atmos_wet = hcho_wet_d 
# else:
#     print('atmos_no is out of bounds (1 to 6)')

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
err_data = []

lc = broadleaf_crop



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
        

    elev_boundary = 1000
    high_elev = ma.masked_greater_equal(dem_crop, elev_boundary).mask

    atmos_dry = mask_high_elev(atmos_dry, high_elev) 
    atmos_wet = mask_high_elev(atmos_wet, high_elev)

    ### keeping only data for months where both datasets are available
    min_time = xr.DataArray(data = [fire_dry.time.min().values, atmos_dry.time.min().values]).max().values
    min_year = min_time.astype(str).split('-')[0]
    max_time = xr.DataArray(data = [fire_dry.time.max().values, atmos_dry.time.max().values]).min().values
    max_year = max_time.astype(str).split('-')[0]

    fire_dry_slice = fire_dry.sel(time=slice(min_year, max_year))
    fire_wet_slice = fire_wet.sel(time=slice(min_year, max_year))
    atmos_dry_slice = atmos_dry.sel(time=slice(min_year, max_year))
    atmos_wet_slice = atmos_wet.sel(time=slice(min_year, max_year))
    lc_slice = lc.sel(time=slice(min_year, max_year))
    reshape_lc = np.zeros_like(atmos_dry_slice)
    for a in range(lc_slice.shape[0]):
        for b in range(3):
            reshape_lc[a*3+b,:,:] = lc_slice[a]

    atmos_wet_1d = atmos_wet_slice.values.ravel()
    atmos_dry_1d = atmos_dry_slice.values.ravel()

    fire_dry_1d = fire_dry_slice.values.ravel()
    fire_wet_1d = fire_wet_slice.values.ravel()
    
    lc_1d = reshape_lc.ravel()


    ### create pandas dataframe to hold data of interest

    data = {'Atmos': atmos_dry_1d,  'Atmos_wet': atmos_wet_1d, 'Fire': fire_dry_1d, 'Fire_wet': fire_wet_1d, 'LC': lc_1d} 

    data_pd = pd.DataFrame(data).replace(-np.Inf, np.nan).dropna()

    minb = 0
    maxb = 0.1
    stepb = 0.01
    
    boundaries_dry = np.arange(minb, maxb, stepb)
    mean_atmos_dry = np.zeros(10)
    std_err_dry = np.zeros(10)

    for i, b in enumerate(boundaries_dry):
        if i == 9:
            subset = data_pd[data_pd['Fire'] >= b]
        else:
            subset = data_pd[data_pd['Fire'] >= b][data_pd['Fire'] < b+stepb]
        print(subset['Atmos'].size)
        mean_atmos_dry[i] = subset['Atmos'].mean()
        std_err_dry[i] = subset['Atmos'].std()/np.sqrt(subset['Atmos'].size)
        
    mean_atmos_for = np.zeros(10)
    std_err_for = np.zeros(10)
    data_for = data_pd[data_pd['LC'] >= 50]

    for i, b in enumerate(boundaries_dry):
        if i == 9:
            subset = data_for[data_for['Fire'] >= b]
        else:
            subset = data_for[data_for['Fire'] >= b][data_for['Fire'] < b+stepb]
        print(subset['Atmos'].size)
        mean_atmos_for[i] = subset['Atmos'].mean()
        std_err_for[i] = subset['Atmos'].std()/np.sqrt(subset['Atmos'].size)

    mean_atmos_sav = np.zeros(10)
    std_err_sav = np.zeros(10)
    data_sav = data_pd[data_pd['LC'] < 50]

    for i, b in enumerate(boundaries_dry):
        if i == 9:
            subset = data_sav[data_sav['Fire'] >= b]
        else:
            subset = data_sav[data_sav['Fire'] >= b][data_sav['Fire'] < b+stepb]
        print(subset['Atmos'].size)
        mean_atmos_sav[i] = subset['Atmos'].mean()
        std_err_sav[i] = subset['Atmos'].std()/np.sqrt(subset['Atmos'].size)
        
        
    boundaries_wet = np.arange(minb, maxb, stepb)
    mean_atmos_wet = np.zeros(10)
    std_err_wet = np.zeros(10)

    for i, b in enumerate(boundaries_wet):
        if i == 9:
            subset = data_pd[data_pd['Fire_wet'] >= b]
        else:
            subset = data_pd[data_pd['Fire_wet'] >= b][data_pd['Fire_wet'] < b+stepb]
        print(subset['Atmos_wet'].size)
        mean_atmos_wet[i] = subset['Atmos_wet'].mean()
        std_err_wet[i] = subset['Atmos_wet'].std()/np.sqrt(subset['Atmos_wet'].size)
    
    data = [mean_atmos_sav, mean_atmos_for, mean_atmos_dry, mean_atmos_wet]
    
    data_err = [std_err_sav, std_err_for, std_err_dry, std_err_wet]
    
    atm_means = np.zeros((4,10))
    err_array = np.zeros((4,10))
    
    for x in range(4):
        atm_means[x,:] = data[x] 
        err_array[x,:] = data_err[x]
    
    plot_data.append(atm_means)
    err_data.append(err_array)

plot_data[0] = plot_data[0]/10**16
plot_data[2] = plot_data[2]/10**16
plot_data[5] = plot_data[5]/10**15

err_data[0] = err_data[0]/10**16
err_data[2] = err_data[2]/10**16
err_data[5] = err_data[5]/10**15


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

labels = {1 : 'Isoprene', 2 :  'Methanol', 3 : 'HCHO', 4 : 'CO', 5 : 'AOD', 6 : 'NO$_{2}$'}

units = {1 : '(10$^{15}$ molecules cm$^{-2}$)', 2 : '(ppbv)', 3 : '(10$^{16}$ molecules cm$^{-2}$)', 4 : '(10$^{17}$ molecules cm$^{-2}$)',\
         5 : 'at 0.47 $\mu$m', 6 : '(10$^{15}$ molecules cm$^{-2}$)'}


alphabet = ['a', 'b', 'c', 'd', 'e', 'f',]
categories = ['S', 'F', 'D', 'W']
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12*cm, 14*cm))
axes = axes.ravel()

for i in range(6):
    sns.heatmap(plot_data[i],  cmap='YlGnBu', ax = axes[i], cbar_kws = {'label':f'{labels[i+1]} {units[i+1]}',\
                                                                        'location':'right'})
    axes[i].text(-2, 0, f'({alphabet[i]})', fontsize = 10)
    axes[i].set_yticklabels(categories, fontsize=8)
    axes[i].set_xlim(0,10)
    axes[i].set_xticks(np.arange(0,11, 5))
    axes[i].set_xticklabels(np.arange(0, maxb+stepb/10, maxb/2), fontsize = 7)
    axes[i].set_xlabel('Burned Area (%)', fontsize = 8)
    cbar = axes[i].collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(f'{labels[i+1]} \n{units[i+1]}', fontsize = 8)


fig.tight_layout()
# fig.subplots_adjust(bottom=0.25)


# save figure
# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/xaxis_fire_wet_dry_for_sav_corrGFED4.png', dpi = 300)
# 

labels = {1 : 'Isoprene std err', 2 :  'Methanol std err', 3 : 'HCHO std err', 4 : 'CO std err', 5 : 'AOD std err', 6 : 'NO$_{2}$ std err'}

units = {1 : '(10$^{15}$ molecules cm$^{-2}$)', 2 : '(ppbv)', 3 : '(10$^{16}$ molecules cm$^{-2}$)', 4 : '(10$^{17}$ molecules cm$^{-2}$)',\
          5 : 'at 0.47 $\mu$m', 6 : '(10$^{15}$ molecules cm$^{-2}$)'}


alphabet = ['a', 'b', 'c', 'd', 'e', 'f',]
categories = ['S', 'F', 'D', 'W']
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12*cm, 14*cm))
axes = axes.ravel()

for i in range(6):
    sns.heatmap(err_data[i],  cmap='YlGnBu', ax = axes[i], cbar_kws = {'label':f'{labels[i+1]} {units[i+1]}',\
                                                                        'location':'right'})
    axes[i].text(-2, 0, f'({alphabet[i]})', fontsize = 10)
    axes[i].set_yticklabels(categories, fontsize=8)
    axes[i].set_xlim(0,10)
    axes[i].set_xticks(np.arange(0,11, 5))
    axes[i].set_xticklabels(np.arange(0, 0.11, 0.05), fontsize = 7)
    axes[i].set_xlabel('Burned Area (%)', fontsize = 8)
    cbar = axes[i].collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(f'{labels[i+1]} \n{units[i+1]}', fontsize = 8)


fig.tight_layout()
# fig.subplots_adjust(bottom=0.25)


# # save figure
# # fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/xaxis_fire_wet_dry_for_sav_stderr.png', dpi = 300)

# # =============================================================================
# # wet season forest divide
# # =============================================================================

# plot_data = []
# err_data = []

# lc = broadleaf_crop



# for i in range(1, 7):
#     atmos_no = i
#     if atmos_no == 1:
#         atmos_dry = isop_dry #aod_dry #no2_dry # co_dry #hcho_dry_d
#         atmos_wet = isop_wet #aod_wet # no2_wet # co_wet #hcho_wet_d
#     elif atmos_no == 2:
#         atmos_dry = methanol_dry
#         atmos_wet = methanol_wet         
#     elif atmos_no == 3:
#         atmos_dry = hcho_dry_d 
#         atmos_wet = hcho_wet_d 
#     elif atmos_no == 4:
#         atmos_dry = co_dry 
#         atmos_wet = co_wet 
#     elif atmos_no == 5:
#         atmos_dry = aod_dry #no2_dry # co_dry #hcho_dry_d
#         atmos_wet = aod_wet # no2_wet # co_wet #hcho_wet_d
#     elif atmos_no == 6:
#         atmos_dry = no2_dry # co_dry #hcho_dry_d
#         atmos_wet = no2_wet # co_wet #hcho_wet_d
#     else:
#         print('atmos_no is out of bounds (1 to 6)')
        

#     elev_boundary = 1000
#     high_elev = ma.masked_greater_equal(dem_crop, elev_boundary).mask

#     atmos_dry = mask_high_elev(atmos_dry, high_elev) 
#     atmos_wet = mask_high_elev(atmos_wet, high_elev)

#     ### keeping only data for months where both datasets are available
#     min_time = xr.DataArray(data = [fire_dry.time.min().values, atmos_dry.time.min().values]).max().values
#     min_year = min_time.astype(str).split('-')[0]
#     max_time = xr.DataArray(data = [fire_dry.time.max().values, atmos_dry.time.max().values]).min().values
#     max_year = max_time.astype(str).split('-')[0]

#     fire_dry_slice = fire_dry.sel(time=slice(min_year, max_year))
#     fire_wet_slice = fire_wet.sel(time=slice(min_year, max_year))
#     atmos_dry_slice = atmos_dry.sel(time=slice(min_year, max_year))
#     atmos_wet_slice = atmos_wet.sel(time=slice(min_year, max_year))
#     lc_slice = lc.sel(time=slice(min_year, max_year))
#     reshape_lc = np.zeros_like(atmos_dry_slice)
#     for a in range(lc_slice.shape[0]):
#         for b in range(3):
#             reshape_lc[a*3+b,:,:] = lc_slice[a]

#     atmos_wet_1d = atmos_wet_slice.values.ravel()
#     atmos_dry_1d = atmos_dry_slice.values.ravel()

#     fire_dry_1d = fire_dry_slice.values.ravel()
#     fire_wet_1d = fire_wet_slice.values.ravel()
    
#     lc_1d = reshape_lc.ravel()


#     ### create pandas dataframe to hold data of interest

#     data = {'Atmos': atmos_dry_1d,  'Atmos_wet': atmos_wet_1d, 'Fire': fire_dry_1d, 'Fire_wet': fire_wet_1d, 'LC': lc_1d} 

#     data_pd = pd.DataFrame(data).replace(-np.Inf, np.nan).dropna()

#     minb = 0
#     maxb = 0.1
#     stepb = 0.1
    
#     boundaries_dry = np.arange(minb, maxb, stepb)
#     mean_atmos_dry = np.zeros(10)
#     std_err_dry = np.zeros(10)

#     for i, b in enumerate(boundaries_dry):
#         if i == 9:
#             subset = data_pd[data_pd['Fire'] >= b]
#         else:
#             subset = data_pd[data_pd['Fire'] >= b][data_pd['Fire'] < b+0.1]
#         print(subset['Atmos'].size)
#         mean_atmos_dry[i] = subset['Atmos'].mean()
#         std_err_dry[i] = subset['Atmos'].std()/np.sqrt(subset['Atmos'].size)
        
#     mean_atmos_for = np.zeros(10)
#     std_err_for = np.zeros(10)
#     data_for = data_pd[data_pd['LC'] >= 50]

#     for i, b in enumerate(boundaries_dry):
#         if i == 9:
#             subset = data_for[data_for['Fire_wet'] >= b]
#         else:
#             subset = data_for[data_for['Fire_wet'] >= b][data_for['Fire_wet'] < b+0.1]
#         print(subset['Atmos_wet'].size)
#         mean_atmos_for[i] = subset['Atmos_wet'].mean()
#         std_err_for[i] = subset['Atmos_wet'].std()/np.sqrt(subset['Atmos_wet'].size)

#     mean_atmos_sav = np.zeros(10)
#     std_err_sav = np.zeros(10)
#     data_sav = data_pd[data_pd['LC'] < 50]

#     for i, b in enumerate(boundaries_dry):
#         if i == 9:
#             subset = data_sav[data_sav['Fire_wet'] >= b]
#         else:
#             subset = data_sav[data_sav['Fire_wet'] >= b][data_sav['Fire_wet'] < b+0.1]
#         print(subset['Atmos_wet'].size)
#         mean_atmos_sav[i] = subset['Atmos_wet'].mean()
#         std_err_sav[i] = subset['Atmos_wet'].std()/np.sqrt(subset['Atmos_wet'].size)
        
        
#     boundaries_wet = np.arange(minb, maxb, stepb)
#     mean_atmos_wet = np.zeros(10)
#     std_err_wet = np.zeros(10)

#     for i, b in enumerate(boundaries_wet):
#         if i == 9:
#             subset = data_pd[data_pd['Fire_wet'] >= b]
#         else:
#             subset = data_pd[data_pd['Fire_wet'] >= b][data_pd['Fire_wet'] < b+0.1]
#         print(subset['Atmos_wet'].size)
#         mean_atmos_wet[i] = subset['Atmos_wet'].mean()
#         std_err_wet[i] = subset['Atmos_wet'].std()/np.sqrt(subset['Atmos_wet'].size)
    
#     data = [mean_atmos_sav, mean_atmos_for, mean_atmos_wet]
    
#     data_err = [std_err_sav, std_err_for, std_err_wet]
    
#     atm_means = np.zeros((3,10))
#     err_array = np.zeros((3,10))
    
#     for x in range(3):
#         atm_means[x,:] = data[x] 
#         err_array[x,:] = data_err[x]
    
#     plot_data.append(atm_means)
#     err_data.append(err_array)

# plot_data[0] = plot_data[0]/10**16
# plot_data[2] = plot_data[2]/10**16
# plot_data[5] = plot_data[5]/10**15

# err_data[0] = err_data[0]/10**16
# err_data[2] = err_data[2]/10**16
# err_data[5] = err_data[5]/10**15


# # # =============================================================================
# # # heatmap plot
# # # =============================================================================
# # categories = ['High Fire', 'Low Fire', 'Dry', 'Wet']

# # categories = ['H', 'L', 'D', 'W']

# # cm = 1/2.54
# # fig, axes = plt.subplots(figsize=(6 * cm, 3 * cm))
# # axes.set_title(labels[atmos_no], fontsize = 8)
# # axes = sns.heatmap(lc_means,  cmap='YlGnBu', cbar_kws = {'label':f'{labels[atmos_no]} {units[atmos_no]}'})#,
# #                                                         # how to change fontsize' : '8' }) #vmin=10, vmax=30,

# # # axes.set_ylabel('Category', fontsize = 10)
# # axes.set_xlabel('Broadleaf forest % cover', fontsize = 8)
# # axes.set_yticklabels(categories, fontsize=8)
# # axes.set_xlim(0,10)
# # axes.set_xticks(np.arange(2,11, 2))
# # axes.set_xticklabels(range(20,101, 20), fontsize = 8)

# # # fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/Summary/{labels[atmos_no]}_heatmap_notweighted_noneabove{elev_boundary}_broadleafB_hldw.png',\
# # #             dpi = 300, bbox_inches='tight')

# # =============================================================================
# # broadleaf heatmaps for all 6 species
# # =============================================================================

# labels = {1 : 'Isoprene', 2 :  'Methanol', 3 : 'HCHO', 4 : 'CO', 5 : 'AOD', 6 : 'NO$_{2}$'}

# units = {1 : '(10$^{15}$ molecules cm$^{-2}$)', 2 : '(ppbv)', 3 : '(10$^{16}$ molecules cm$^{-2}$)', 4 : '(10$^{17}$ molecules cm$^{-2}$)',\
#          5 : 'at 0.47 $\mu$m', 6 : '(10$^{15}$ molecules cm$^{-2}$)'}


# alphabet = ['a', 'b', 'c', 'd', 'e', 'f',]
# categories = ['Sw', 'Fw', 'W']
# cm = 1/2.54
# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12*cm, 14*cm))
# axes = axes.ravel()

# for i in range(6):
#     sns.heatmap(plot_data[i],  cmap='YlGnBu', ax = axes[i], cbar_kws = {'label':f'{labels[i+1]} {units[i+1]}',\
#                                                                         'location':'right'})
#     axes[i].text(-2, 0, f'({alphabet[i]})', fontsize = 10)
#     axes[i].set_yticklabels(categories, fontsize=8)
#     axes[i].set_xlim(0,10)
#     axes[i].set_xticks(np.arange(0,11, 5))
#     axes[i].set_xticklabels(np.arange(0, 1.1, 0.5), fontsize = 7)
#     axes[i].set_xlabel('Burned Area (%)', fontsize = 8)
#     cbar = axes[i].collections[0].colorbar
#     cbar.ax.tick_params(labelsize=7)
#     cbar.set_label(f'{labels[i+1]} \n{units[i+1]}', fontsize = 8)


# fig.tight_layout()
# # fig.subplots_adjust(bottom=0.25)


# # save figure
# # fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/xaxis_fire_wet_for_sav_wetseason.png', dpi = 300)


# # labels = {1 : 'Isoprene std err', 2 :  'Methanol std err', 3 : 'HCHO std err', 4 : 'CO std err', 5 : 'AOD std err', 6 : 'NO$_{2}$ std err'}

# # units = {1 : '(10$^{15}$ molecules cm$^{-2}$)', 2 : '(ppbv)', 3 : '(10$^{16}$ molecules cm$^{-2}$)', 4 : '(10$^{17}$ molecules cm$^{-2}$)',\
# #          5 : 'at 0.47 $\mu$m', 6 : '(10$^{15}$ molecules cm$^{-2}$)'}


# # alphabet = ['a', 'b', 'c', 'd', 'e', 'f',]
# # categories = ['Sw', 'Fw', 'W']
# # cm = 1/2.54
# # fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12*cm, 14*cm))
# # axes = axes.ravel()

# # for i in range(6):
# #     sns.heatmap(err_data[i],  cmap='YlGnBu', ax = axes[i], cbar_kws = {'label':f'{labels[i+1]} {units[i+1]}',\
# #                                                                         'location':'right'})
# #     axes[i].text(-2, 0, f'({alphabet[i]})', fontsize = 10)
# #     axes[i].set_yticklabels(categories, fontsize=8)
# #     axes[i].set_xlim(0,10)
# #     axes[i].set_xticks(np.arange(0,11, 5))
# #     axes[i].set_xticklabels(np.arange(0, 1.1, 0.5), fontsize = 7)
# #     axes[i].set_xlabel('Burned Area (%)', fontsize = 8)
# #     cbar = axes[i].collections[0].colorbar
# #     cbar.ax.tick_params(labelsize=7)
# #     cbar.set_label(f'{labels[i+1]} \n{units[i+1]}', fontsize = 8)


# # fig.tight_layout()
# # # fig.subplots_adjust(bottom=0.25)


# # # save figure
# # # fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/xaxis_fire_wet_for_sav_wetseason_stderr.png', dpi = 300)
