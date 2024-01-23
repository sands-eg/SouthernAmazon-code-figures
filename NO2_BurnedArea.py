# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:28:52 2023

@author: s2261807

Script for analysis of NO2 ~ burned area relationship in the southern Amazon.

Focus on understanding the different relationships at different vegetation covers.

"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy.ma as ma
from sklearn.linear_model import TheilSenRegressor
import statsmodels.api as sm
from scipy.stats import spearmanr

# =============================================================================
# load data
# =============================================================================

# burned area (GFED 4; 2001-2016); error in metadata - unit is fraction of cell, not %
fn = 'R:\gfed\GFED5\GFED5_totalBA_2001-2020.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire = ds['__xarray_dataarray_variable__']
fire['time'] = pd.date_range('2001-01-01', '2020-12-31', freq = 'MS')
ds.close()

## DEM
fn = 'R:/DEM/SAM_DEM.nc'
ds = xr.open_dataset(fn)
dem = ds['SAM_DEM']
ds.close() 

### LAI
fn = 'R:\modis_lai\modis_lai_2001-2019.nc'
ds = xr.open_dataset(fn, decode_times=False)
lai = ds['LAI']#[:,::-1,:]
lai['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()

### broadleaf forest
fn = 'R:\modis_lc\mcd12c1_1deg_igbp_percent_corrected.nc'
ds = xr.open_dataset(fn, decode_times=True)
broadleaf = (ds['Land_Cover_Type_1_Percent'][:,:,:,2] + ds['Land_Cover_Type_1_Percent'][:,:,:,4])
broadleaf['time'] = pd.date_range('2001', '2020', freq = 'Y')
ds.close()

### NO2
fn = 'R:/OMI_NO2/omi_no2_mm_2005_2020_masked.nc'
ds = xr.open_dataset(fn, decode_times=True)
no2 = ds['no2'][:,:,:]
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
# initialise variables
lon_edges = np.arange(-180, 180.1, 1)
lat_edges = np.arange(90, -90.1, -1)
n_lons = len(lon_edges) - 1
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
# crop data to region of interest
# =============================================================================
fire = fire.fillna(0)/surface_area_earth *100
# fire = fire/surface_area_earth *100
fire_crop = crop_data(fire)
lai_crop = crop_data(lai)
broadleaf_crop = crop_data(broadleaf)
dem_crop = crop_data(dem)

no2_crop = crop_data(no2)

# =============================================================================
# separate data into wet and dry season
# =============================================================================
fire_wet = get_month_data([2,3,4], fire_crop)
fire_dry = get_month_data([8,9,10], fire_crop)

no2_wet = get_month_data([2,3,4], no2_crop)
no2_dry = get_month_data([8,9,10], no2_crop)


# =============================================================================
# create high-fire and low-fire data subsets = mask my burned area - what value should be chosen? - start with any fire in a given month
# =============================================================================
atmos_no = 4
atmos_dry = no2_dry 
atmos_wet = no2_wet 


labels = {1 : 'Isoprene', 2 : 'AOD' , 3 : 'Methanol', 4 : 'NO$_{2}$', 5 : 'CO', 6 : 'HCHO'}

units = {1 : '10$^{16}$ molecules cm$^{-2}$', 2 : 'at 0.47 $\mu$m', 3 : '[ppbv]', 4 : '[molecules cm$^{-2}$]',\
         5 : '[10$^{17}$ molecules cm$^{-2}$]', 6 : '[molecules cm$^{-2}$]'}


lc = broadleaf_crop
lc_label = 'Broadleaf Forest'

elev_boundary = 1000
high_elev = ma.masked_greater_equal(dem_crop, elev_boundary).mask

atmos_dry = mask_high_elev(atmos_dry, high_elev) 
atmos_wet = mask_high_elev(atmos_wet, high_elev)

# ### keeping only data for months where both datasets are available
min_time = xr.DataArray(data = [fire_dry.time.min().values, atmos_dry.time.min().values]).max().values
min_year = min_time.astype(str).split('-')[0]
max_time = xr.DataArray(data = [fire_dry.time.max().values, atmos_dry.time.max().values]).min().values
max_year = max_time.astype(str).split('-')[0]


fire_slice = fire_dry.sel(time=slice(min_year, max_year))
fire_wet_slice = fire_wet.sel(time=slice(min_year, max_year))
atmos_dry_slice = atmos_dry.sel(time=slice(min_year, max_year))
atmos_wet_slice = atmos_wet.sel(time=slice(min_year, max_year))
lc_slice = lc.sel(time=slice(min_year, max_year))
lai_slice = lai.sel(time=slice(min_year, max_year))
lc_int = np.rint(lc_slice)

# ### masking fire occurrence
# boundary = 0.004
# no_fire = ma.masked_less_equal(fire_slice, boundary).mask
# yes_fire = ma.masked_greater(fire_slice, boundary).mask
    
# atmos_dry_fire, atmos_dry_no_fire = mask_data(atmos_dry_slice, yes_fire, no_fire)

reshape_lc = np.zeros_like(atmos_dry_slice)
for a in range(lc_int.shape[0]):
    for b in range(3):
        reshape_lc[a*3+b,:,:] = lc_int[a]

reshape_lai = np.zeros_like(atmos_dry_slice)
for a in range(lai_slice.shape[0]):
    for b in range(3):
        reshape_lai[a*3+b,:,:] = lai_slice[a]
        
# =============================================================================
# get lat and lon as separate arrays
# =============================================================================
lat_array = np.zeros((20,20))
for i in range(20):
    lat_array[:,i] = atmos_dry_slice.lat[:]

lon_array = np.zeros((20,20))
for i in range(20):
    lon_array[i,:] = atmos_dry_slice.lon[:]

reshape_lat = np.zeros_like(atmos_dry_slice)
for a in range(reshape_lat.shape[0]):
    reshape_lat[a,:,:] = lat_array[:,:]

reshape_lon = np.zeros_like(atmos_dry_slice)
for a in range(reshape_lon.shape[0]):
    reshape_lon[a,:,:] = lon_array[:,:]
    
# =============================================================================
# Create dataframe - create 1 D lat and lon variables
# =============================================================================
atmos_wet_1d = atmos_wet_slice.values.ravel()
atmos_dry_1d = atmos_dry_slice.values.ravel()
lai_1d = reshape_lai.ravel()
lc_1d = reshape_lc.ravel()
fire_1d = fire_slice.values.ravel()
fire_wet_1d = fire_wet_slice.values.ravel()
lat_1d = reshape_lat.ravel()
lon_1d = reshape_lon.ravel()

### create pandas dataframe to hold data of interest

data = {'Atmos': atmos_dry_1d,  'Atmos_wet': atmos_wet_1d, 'LC': lc_1d, 'LAI': lai_1d, 'Fire': fire_1d, 'Fire_wet': fire_wet_1d,\
        'Lat' : lat_1d, 'Lon' : lon_1d} 

data_log = {'Atmos': atmos_dry_1d,  'Atmos_wet': atmos_wet_1d, 'LC': lc_1d, 'LAI': lai_1d, 'Fire': fire_1d, 'Fire_wet': fire_wet_1d,\
        'Atmos_log': np.log(atmos_dry_1d), 'Atmos_sqrt': np.sqrt(atmos_dry_1d),\
            'Fire_log': np.log(fire_1d), 'Fire_wet_log': np.log(fire_wet_1d), 'Fire_sqrt': np.sqrt(fire_1d), 'Lat' : lat_1d, 'Lon' : lon_1d} #, 'expFire': np.exp(fire_1d)}
    
data_pd = pd.DataFrame(data).replace(-np.Inf, np.nan).dropna()
data_log_pd = pd.DataFrame(data_log).replace(-np.Inf, np.nan).dropna()


# =============================================================================
# All data regression: OLS and Theil-Sen
# =============================================================================
X = data_pd['Fire'].values.reshape(-1, 1)
y =data_pd['Atmos'].values
# Theil-Sen
model = TheilSenRegressor(fit_intercept=True)
reg = model.fit(X, y)
slope = reg.coef_
inter = reg.intercept_
R = reg.score(X, y)

plt.scatter(X, y, s=2)
plt.plot(X, X*slope+inter, lw=1, c='grey')
print(R)

Xi_all = data_pd['Fire']
# Xi = data_pd[['LC', 'Fire']]
y_all = data_pd['Atmos']

X_all = sm.add_constant(Xi_all)

reg = TheilSenRegressor().fit(X_all, y_all)
score = reg.score(X_all, y_all)
print(score)
y_pred_all = reg.predict(X_all)


model = sm.OLS(y_all, X_all)
results = model.fit()
print(results.summary())

plt.scatter(Xi_all, y_all, s=2, c='grey')
plt.plot(Xi_all, y_pred_all, lw=1, c='blue')
plt.plot(Xi_all, results.params[0] + Xi_all*results.params[1], lw=1, c='red')

X = data_log_pd['Fire_log'].values.reshape(-1, 1)
y = data_log_pd['Atmos_log'].values
# Theil-Sen
model = TheilSenRegressor(fit_intercept=True)
reg = model.fit(X, y)
slope = reg.coef_
inter = reg.intercept_
R = reg.score(X, y)

plt.scatter(X, y, s=2)
plt.plot(X, X*slope+inter, lw=1, c='grey')
print(R)

Xi_all = data_log_pd['Fire']
# Xi = data_pd[['LC', 'Fire']]
y_all = data_log_pd['Atmos']

X_all = sm.add_constant(Xi_all)

reg = TheilSenRegressor().fit(X_all, y_all)
# reg = OLS().fit(X, y)

score = reg.score(X_all, y_all)
print(score)
y_pred_all = reg.predict(X_all)
plt.scatter(Xi_all, y_all, s=2)
plt.plot(Xi_all, y_pred_all, lw=1, c='grey')
# =============================================================================
# NO2 figure - points coloured based on lai, lc, lat, lon
# =============================================================================

### Original dry season data
cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
axes = axes.ravel()

a = axes[0].scatter(data_pd['Fire'], data_pd['Atmos'], s=2, c=data_pd['LAI']/10, cmap='YlGnBu')
fig.colorbar(a,ax=axes[0],orientation='vertical',label='LAI')

b = axes[1].scatter(data_pd['Fire'], data_pd['Atmos'], s=2, c=data_pd['LC'], cmap='YlGnBu')
fig.colorbar(b,ax=axes[1],orientation='vertical',label='Forest Cover')
c = axes[2].scatter(data_pd['Fire'], data_pd['Atmos'], s=2, c=data_pd['Lat'], cmap='YlGnBu')
fig.colorbar(c,ax=axes[2],orientation='vertical',label='Latitude')
d = axes[3].scatter(data_pd['Fire'], data_pd['Atmos'], s=2, c=data_pd['Lon'], cmap='YlGnBu')
fig.colorbar(d,ax=axes[3],orientation='vertical',label='Longitude')
for i in range(4):
    axes[i].set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 10)
    axes[i].set_xlabel('Burned area [% grid cell area]', size = 10)
    
fig.tight_layout()

# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_dry_season_coloured_scatter_gfed5.png', dpi = 300)

### Original wet season data
cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
axes = axes.ravel()

a = axes[0].scatter(data_pd['Fire_wet'], data_pd['Atmos_wet'], s=2, c=data_pd['LAI']/10, cmap='YlGnBu')
fig.colorbar(a,ax=axes[0],orientation='vertical',label='LAI')

b = axes[1].scatter(data_pd['Fire_wet'], data_pd['Atmos_wet'], s=2, c=data_pd['LC'], cmap='YlGnBu')
fig.colorbar(b,ax=axes[1],orientation='vertical',label='Forest Cover')
c = axes[2].scatter(data_pd['Fire_wet'], data_pd['Atmos_wet'], s=2, c=data_pd['Lat'], cmap='YlGnBu')
fig.colorbar(c,ax=axes[2],orientation='vertical',label='Latitude')
d = axes[3].scatter(data_pd['Fire_wet'], data_pd['Atmos_wet'], s=2, c=data_pd['Lon'], cmap='YlGnBu')
fig.colorbar(d,ax=axes[3],orientation='vertical',label='Longitude')
for i in range(4):
    axes[i].set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 10)
    axes[i].set_xlabel('Burned area [% grid cell area]', size = 10)
    axes[i].set_xlim([0, 0.002*100])
    
fig.tight_layout()

# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_wet_season_coloured_scatter_gfed5.png', dpi = 300)


### Natural logarithm, dry season data
cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
axes = axes.ravel()

a = axes[0].scatter(data_log_pd['Fire_log'], data_log_pd['Atmos_log'], s=2, c=data_log_pd['LAI']/10, cmap='YlGnBu')
fig.colorbar(a,ax=axes[0],orientation='vertical',label='LAI')

b = axes[1].scatter(data_log_pd['Fire_log'], data_log_pd['Atmos_log'], s=2, c=data_log_pd['LC'], cmap='YlGnBu')
fig.colorbar(b,ax=axes[1],orientation='vertical',label='Forest Cover')
c = axes[2].scatter(data_log_pd['Fire_log'], data_log_pd['Atmos_log'], s=2, c=data_log_pd['Lat'], cmap='YlGnBu')
fig.colorbar(c,ax=axes[2],orientation='vertical',label='Latitude')
d = axes[3].scatter(data_log_pd['Fire_log'], data_log_pd['Atmos_log'], s=2, c=data_log_pd['Lon'], cmap='YlGnBu')
fig.colorbar(d,ax=axes[3],orientation='vertical',label='Longitude')
for i in range(4):
    axes[i].set_ylabel(f'Nat log of {labels[atmos_no]} {units[atmos_no]}', size = 10)
    axes[i].set_xlabel('Nat log of burned area \n[% grid cell area]', size = 10)
    
fig.tight_layout()

# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/log{labels[atmos_no]}_dry_season_coloured_scatter_gfed5.png', dpi = 300)


# =============================================================================
# focusing on forest cover role
# =============================================================================

def OLS_regression(Xi, y):
    X = sm.add_constant(Xi)
    y = y
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(res.summary())
    y_pred = res.predict(X)

    plt.figure()
    plt.scatter(Xi, y, color="grey", s=2, label = 'Data')
    plt.plot(Xi, y_pred, label = 'Predicted')
    
    return res

def TS_regression(Xi, y):
    X = sm.add_constant(Xi)
    y = y
    res = TheilSenRegressor().fit(X, y)
    print(res.score(X, y))
    y_pred = res.predict(X)

    plt.figure()
    plt.scatter(Xi, y, color="grey", s=2, label = 'Data')
    plt.plot(Xi, y_pred, label = 'Predicted')
    return res



forest75_pd = data_pd[data_pd['LC'] > 75]
print(len(forest75_pd))
res75 = OLS_regression(forest75_pd['Fire'], forest75_pd['Atmos'])
ts75 = TS_regression(forest75_pd['Fire'], forest75_pd['Atmos'])
forest50_pd = data_pd[data_pd['LC'] <= 75][data_pd['LC'] > 50]
print(len(forest50_pd))
res50 = OLS_regression(forest50_pd['Fire'], forest50_pd['Atmos'])
ts50 = TS_regression(forest50_pd['Fire'], forest50_pd['Atmos'])
forest25_pd = data_pd[data_pd['LC'] <= 50][data_pd['LC'] > 25]
print(len(forest25_pd))
res25 = OLS_regression(forest25_pd['Fire'], forest25_pd['Atmos'])
ts25 = TS_regression(forest25_pd['Fire'], forest25_pd['Atmos'])
forest0_pd = data_pd[data_pd['LC'] <= 25]
print(len(forest0_pd))
res0 = OLS_regression(forest0_pd['Fire'], forest0_pd['Atmos'])
ts0 = TS_regression(forest0_pd['Fire'], forest0_pd['Atmos'])


### Original data split into 4 classes
cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
axes = axes.ravel()

a = axes[0].scatter(forest75_pd['Fire'], forest75_pd['Atmos'], s=2, c=forest75_pd['LC'], cmap='YlGnBu')
axes[0].plot(forest75_pd['Fire'], forest75_pd['Fire']*res75.params[1]+res75.params[0], c ='grey')
axes[0].plot(forest75_pd['Fire'], ts75.predict(sm.add_constant(forest75_pd['Fire'])), c ='red')
fig.colorbar(a,ax=axes[0],orientation='vertical',label='Forest Cover')
b = axes[1].scatter(forest50_pd['Fire'], forest50_pd['Atmos'], s=2, c=forest50_pd['LC'], cmap='YlGnBu')
axes[1].plot(forest50_pd['Fire'], forest50_pd['Fire']*res50.params[1]+res50.params[0], c ='grey')
axes[1].plot(forest50_pd['Fire'], ts50.predict(sm.add_constant(forest50_pd['Fire'])), c ='red')
fig.colorbar(b,ax=axes[1],orientation='vertical',label='Forest Cover')
c = axes[2].scatter(forest25_pd['Fire'], forest25_pd['Atmos'], s=2, c=forest25_pd['LC'], cmap='YlGnBu')
axes[2].plot(forest25_pd['Fire'], forest25_pd['Fire']*res25.params[1]+res25.params[0], c ='grey')
axes[2].plot(forest25_pd['Fire'], ts25.predict(sm.add_constant(forest25_pd['Fire'])), c ='red')
fig.colorbar(c,ax=axes[2],orientation='vertical',label='Forest Cover')
d = axes[3].scatter(forest0_pd['Fire'], forest0_pd['Atmos'], s=2, c=forest0_pd['LC'], cmap='YlGnBu')
axes[3].plot(forest0_pd['Fire'], forest0_pd['Fire']*res0.params[1]+res0.params[0], c ='grey')
axes[3].plot(forest0_pd['Fire'], ts0.predict(sm.add_constant(forest0_pd['Fire'])), c ='red')
fig.colorbar(d,ax=axes[3],orientation='vertical',label='Forest Cover')

for i in range(4):
    axes[i].set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 10)
    axes[i].set_xlabel('Burned area [% grid cell area]', size = 10)
    # axes[i].set_xlim([0,1])
    axes[i].set_ylim([0,1 * 10**16])
axes[0].set_xlim([0,0.5])
axes[1].set_xlim([0,0.5])
axes[2].set_xlim([0,0.75])
axes[3].set_xlim([0,1])
# axes[0].set_xlim([0,0.02])
fig.tight_layout()

# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_dry_season_forest_dependency_gfed5_reg.png', dpi = 300)
# 

### natural logarithm, data split into 4 forest cover classes
forest75_pd_log = data_log_pd[data_log_pd['LC'] > 75]
forest50_pd_log = data_log_pd[data_log_pd['LC'] <= 75][data_log_pd['LC'] > 50]
forest25_pd_log = data_log_pd[data_log_pd['LC'] <= 50][data_log_pd['LC'] > 25]
forest0_pd_log = data_log_pd[data_log_pd['LC'] <= 25]

cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
axes = axes.ravel()

a = axes[0].scatter(forest75_pd_log['Fire_log'], forest75_pd_log['Atmos_log'], s=2, c=forest75_pd_log['LC'], cmap='YlGnBu')
fig.colorbar(a,ax=axes[0],orientation='vertical',label='Forest Cover')
b = axes[1].scatter(forest50_pd_log['Fire_log'], forest50_pd_log['Atmos_log'], s=2, c=forest50_pd_log['LC'], cmap='YlGnBu')
fig.colorbar(b,ax=axes[1],orientation='vertical',label='Forest Cover')
c = axes[2].scatter(forest25_pd_log['Fire_log'], forest25_pd_log['Atmos_log'], s=2, c=forest25_pd_log['LC'], cmap='YlGnBu')
fig.colorbar(c,ax=axes[2],orientation='vertical',label='Forest Cover')
d = axes[3].scatter(forest0_pd_log['Fire_log'], forest0_pd_log['Atmos_log'], s=2, c=forest0_pd_log['LC'], cmap='YlGnBu')
fig.colorbar(d,ax=axes[3],orientation='vertical',label='Forest Cover')
for i in range(4):
    axes[i].set_ylabel(f'Log {labels[atmos_no]} {units[atmos_no]}', size = 10)
    axes[i].set_xlabel('Log Burned area [% grid cell area]', size = 10)
    # axes[i].set_xlim([-10, -2.5])
    axes[i].set_ylim([33, 38])
fig.tight_layout()

# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/log{labels[atmos_no]}_dry_season_forest_dependency_gfed5.png', dpi = 300)


forest75_pd = data_pd[data_pd['LC'] > 75]
print(len(forest75_pd))
res75 = OLS_regression(forest75_pd['Fire_wet'], forest75_pd['Atmos_wet'])
ts75 = TS_regression(forest75_pd['Fire_wet'], forest75_pd['Atmos_wet'])
forest50_pd = data_pd[data_pd['LC'] <= 75][data_pd['LC'] > 50]
print(len(forest50_pd))
res50 = OLS_regression(forest50_pd['Fire_wet'], forest50_pd['Atmos_wet'])
ts50 = TS_regression(forest50_pd['Fire_wet'], forest50_pd['Atmos_wet'])
forest25_pd = data_pd[data_pd['LC'] <= 50][data_pd['LC'] > 25]
print(len(forest25_pd))
res25 = OLS_regression(forest25_pd['Fire_wet'], forest25_pd['Atmos_wet'])
ts25 = TS_regression(forest25_pd['Fire_wet'], forest25_pd['Atmos_wet'])
forest0_pd = data_pd[data_pd['LC'] <= 25]
print(len(forest0_pd))
res0 = OLS_regression(forest0_pd['Fire_wet'], forest0_pd['Atmos_wet'])
ts0 = TS_regression(forest0_pd['Fire_wet'], forest0_pd['Atmos_wet'])


### Original data split into 4 classes
cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
axes = axes.ravel()

a = axes[0].scatter(forest75_pd['Fire_wet'], forest75_pd['Atmos_wet'], s=2, c=forest75_pd['LC'], cmap='YlGnBu')
axes[0].plot(forest75_pd['Fire_wet'], forest75_pd['Fire_wet']*res75.params[1]+res75.params[0], c ='grey')
axes[0].plot(forest75_pd['Fire_wet'], ts75.predict(sm.add_constant(forest75_pd['Fire_wet'])), c ='red')
fig.colorbar(a,ax=axes[0],orientation='vertical',label='Forest Cover')
b = axes[1].scatter(forest50_pd['Fire_wet'], forest50_pd['Atmos_wet'], s=2, c=forest50_pd['LC'], cmap='YlGnBu')
axes[1].plot(forest50_pd['Fire_wet'], forest50_pd['Fire_wet']*res50.params[1]+res50.params[0], c ='grey')
axes[1].plot(forest50_pd['Fire_wet'], ts50.predict(sm.add_constant(forest50_pd['Fire_wet'])), c ='red')
fig.colorbar(b,ax=axes[1],orientation='vertical',label='Forest Cover')
c = axes[2].scatter(forest25_pd['Fire_wet'], forest25_pd['Atmos_wet'], s=2, c=forest25_pd['LC'], cmap='YlGnBu')
axes[2].plot(forest25_pd['Fire_wet'], forest25_pd['Fire_wet']*res25.params[1]+res25.params[0], c ='grey')
axes[2].plot(forest25_pd['Fire_wet'], ts25.predict(sm.add_constant(forest25_pd['Fire_wet'])), c ='red')
fig.colorbar(c,ax=axes[2],orientation='vertical',label='Forest Cover')
d = axes[3].scatter(forest0_pd['Fire_wet'], forest0_pd['Atmos_wet'], s=2, c=forest0_pd['LC'], cmap='YlGnBu')
axes[3].plot(forest0_pd['Fire_wet'], forest0_pd['Fire_wet']*res0.params[1]+res0.params[0], c ='grey')
axes[3].plot(forest0_pd['Fire_wet'], ts0.predict(sm.add_constant(forest0_pd['Fire_wet'])), c ='red')
fig.colorbar(d,ax=axes[3],orientation='vertical',label='Forest Cover')

for i in range(4):
    axes[i].set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 10)
    axes[i].set_xlabel('Burned area [% grid cell area]', size = 10)
    axes[i].set_xlim([0,0.2])
    axes[i].set_ylim([0, 2 * 10**15])
# axes[0].set_xlim([0,0.5])
# axes[1].set_xlim([0,0.5])
# axes[2].set_xlim([0,0.75])
# axes[3].set_xlim([0,1])
# axes[0].set_xlim([0,0.02])
fig.tight_layout()

# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_dry_season_forest_dependency_gfed5_reg.png', dpi = 300)
# 
# =============================================================================
# ### splitting data into 10 classes
# =============================================================================

### spearman's correlation dry season
spear_corr = np.zeros((10,2))
boundaries = np.arange(10, 100, 10)
mean_veg = np.zeros(10)

forest90_pd = data_pd[data_pd['LC'] >= 90]
r90 = spearmanr(forest90_pd['Fire'], forest90_pd['Atmos'])
mean_veg[-1] = forest90_pd['LC'].mean()

for i, b in enumerate(boundaries):
    subset = data_pd[data_pd['LC'] < b][data_pd['LC'] >= b-10]
    print(subset['Atmos'].size)
    spear_corr[i,0] = spearmanr(subset['Fire'], subset['Atmos'])[0]
    spear_corr[i,1] = spearmanr(subset['Fire'], subset['Atmos'])[1]   
    mean_veg[i] = subset['LC'].mean()

spear_corr[-1,:] = r90[:]
# spear_corr[:,1].max()

### Figure displaying change in spearman correlation with forest cover bin
cm = 1/2.54
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18*cm, 14*cm))

axes.scatter(mean_veg, spear_corr[:,0], s=5)
axes.set_ylabel(f"Spearman's r between {labels[atmos_no]} and burned area", size = 10)
axes.set_xlabel("Mean forest cover % in bin", size = 10)
axes.set_xlim([0, 100])
axes.set_ylim([0.3, 0.5])
axes.set_title('Dry season')
fig.tight_layout()

### spearman's correlation wet season
spear_corr = np.zeros((10,2))
boundaries = np.arange(10, 100, 10)
mean_veg = np.zeros(10)

forest90_pd = data_pd[data_pd['LC'] >= 90]
r90 = spearmanr(forest90_pd['Fire_wet'], forest90_pd['Atmos_wet'])
mean_veg[-1] = forest90_pd['LC'].mean()

for i, b in enumerate(boundaries):
    subset = data_pd[data_pd['LC'] < b][data_pd['LC'] >= b-10]
    print(subset['Atmos_wet'].size)
    spear_corr[i,0] = spearmanr(subset['Fire_wet'], subset['Atmos_wet'])[0]
    spear_corr[i,1] = spearmanr(subset['Fire_wet'], subset['Atmos_wet'])[1]   
    mean_veg[i] = subset['LC'].mean()

spear_corr[-1,:] = r90[:]
# spear_corr[:,1].max()

### Figure displaying change in spearman correlation with forest cover bin
cm = 1/2.54
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18*cm, 14*cm))

axes.scatter(mean_veg, spear_corr[:,0], s=5)
axes.set_ylabel(f"Spearman's r between {labels[atmos_no]} and burned area", size = 10)
axes.set_xlabel("Mean forest cover % in bin", size = 10)
axes.set_xlim([0, 100])
axes.set_title('Wet season')
fig.tight_layout()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### NO2 : Burned Area ratio dry season
ratio = np.zeros(10)
boundaries = np.arange(10, 100, 10)
mean_veg = np.zeros(10)

forest90_pd = data_pd[data_pd['LC'] >= 90]
rat90 = forest90_pd['Atmos'].mean() / forest90_pd['Fire'].mean()
mean_veg[-1] = forest90_pd['LC'].mean()
ratio[-1] = rat90

for i, b in enumerate(boundaries):
    subset = data_pd[data_pd['LC'] < b][data_pd['LC'] >= b-10]
    print(subset['Atmos'].size)
    ratio[i] = subset['Atmos'].mean() / subset['Fire'].mean()
    mean_veg[i] = subset['LC'].mean()

# Figure displaying change in NO2:burned area ratio with forest cover bin
cm = 1/2.54
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18*cm, 14*cm))

axes.scatter(mean_veg, ratio, s=5)
axes.set_ylabel(f"{labels[atmos_no]} : burned area", size = 10)
axes.set_xlabel("Mean forest cover % in bin", size = 10)
axes.set_xlim([0, 100])
# axes.set_ylim([0.4, 0.8])
axes.set_title('Dry season')
fig.tight_layout()

### NO2 : Burned Area ratio wet season
ratio = np.zeros(10)
boundaries = np.arange(10, 100, 10)
mean_veg = np.zeros(10)

forest90_pd = data_pd[data_pd['LC'] >= 90]
rat90 = forest90_pd['Atmos_wet'].mean() / forest90_pd['Fire_wet'].mean()
mean_veg[-1] = forest90_pd['LC'].mean()
ratio[-1] = rat90

for i, b in enumerate(boundaries):
    subset = data_pd[data_pd['LC'] < b][data_pd['LC'] >= b-10]
    print(subset['Atmos_wet'].size)
    ratio[i] = subset['Atmos_wet'].mean() / subset['Fire_wet'].mean()
    mean_veg[i] = subset['LC'].mean()

# Figure displaying change in NO2:burned area ratio with forest cover bin
cm = 1/2.54
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18*cm, 14*cm))

axes.scatter(mean_veg, ratio, s=5)
axes.set_ylabel(f"{labels[atmos_no]} : burned area", size = 10)
axes.set_xlabel("Mean forest cover % in bin", size = 10)
axes.set_xlim([0, 100])
# axes.set_ylim([0.4, 0.8])
axes.set_title('Wet season')
fig.tight_layout()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### regression slope
slope = np.zeros((10,2))
boundaries = np.arange(10, 100, 10)
mean_veg = np.zeros(10)

forest90_pd = data_pd[data_pd['LC'] >= 90]
mean_veg[-1] = forest90_pd['LC'].mean()

X = forest90_pd['Fire'].values.reshape(-1, 1)
y = forest90_pd['Atmos'].values
# Theil-Sen
model = TheilSenRegressor(fit_intercept=True)
reg = model.fit(X, y)
slope[-1,0] = reg.coef_
inter = reg.intercept_
slope[-1,1] = reg.score(X, y)

plt.scatter(X, y, s=2)
plt.plot(X, X*slope[-1,0]+slope[-1,1], lw=1, c='grey')

for i, b in enumerate(boundaries):
    subset = data_pd[data_pd['LC'] < b][data_pd['LC'] >= b-10]
    X = subset['Fire'].values.reshape(-1, 1)
    y = subset['Atmos'].values
    model = TheilSenRegressor(fit_intercept=True)
    reg = model.fit(X, y)
    slope[i,0] = reg.coef_
    inter = reg.intercept_
    slope[i,1] = reg.score(X, y)
    mean_veg[i] = subset['LC'].mean()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18*cm, 14*cm))
    axes.scatter(X, y, s=2)
    axes.plot(X, X*slope[-1,0]+inter, lw=1, c='grey')

# Figure displaying change in NO2:burned area ratio with forest cover bin
cm = 1/2.54
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18*cm, 14*cm))

axes.scatter(mean_veg, slope[:,0], s=5)
ax2 = axes.twinx()
ax2.plot(mean_veg, slope[:,1])
axes.set_ylabel(f"Slope of {labels[atmos_no]} ~ burned area regression", size = 10)
axes.set_xlabel("Mean forest cover % in bin", size = 10)
axes.set_xlim([0, 100])
# axes.set_ylim([0.4, 0.8])
axes.set_title('Dry season')
fig.tight_layout()


### regression slope wet season
slope = np.zeros((10,2))
boundaries = np.arange(10, 100, 10)
mean_veg = np.zeros(10)

forest90_pd = data_pd[data_pd['LC'] >= 90]
mean_veg[-1] = forest90_pd['LC'].mean()

X = forest90_pd['Fire_wet'].values.reshape(-1, 1)
y = forest90_pd['Atmos_wet'].values
# Theil-Sen
model = TheilSenRegressor(fit_intercept=True)
reg = model.fit(X, y)
slope[-1,0] = reg.coef_
inter = reg.intercept_
slope[-1,1] = reg.score(X, y)

plt.scatter(X, y, s=2)
plt.plot(X, X*slope[-1,0]+slope[-1,1], lw=1, c='grey')

for i, b in enumerate(boundaries):
    subset = data_pd[data_pd['LC'] < b][data_pd['LC'] >= b-10]
    X = subset['Fire_wet'].values.reshape(-1, 1)
    y = subset['Atmos_wet'].values
    model = TheilSenRegressor(fit_intercept=True)
    reg = model.fit(X, y)
    slope[i,0] = reg.coef_
    inter = reg.intercept_
    slope[i,1] = reg.score(X, y)
    mean_veg[i] = subset['LC'].mean()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18*cm, 14*cm))
    axes.scatter(X, y, s=2)
    axes.plot(X, X*slope[-1,0]+inter, lw=1, c='grey')

# Figure displaying change in NO2:burned area ratio with forest cover bin
cm = 1/2.54
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18*cm, 14*cm))

axes.scatter(mean_veg, slope[:,0], s=5)
ax2 = axes.twinx()
ax2.plot(mean_veg, slope[:,1])
axes.set_ylabel(f"Slope of {labels[atmos_no]} ~ burned area regression", size = 10)
axes.set_xlabel("Mean forest cover % in bin", size = 10)
axes.set_xlim([0, 100])
# axes.set_ylim([0.4, 0.8])
axes.set_title('Wet season')
fig.tight_layout()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### regression slope - log data
slope = np.zeros((10,2))
boundaries = np.arange(10, 100, 10)
mean_veg = np.zeros(10)

forest90_pd = data_pd[data_pd['LC'] >= 90]
mean_veg[-1] = forest90_pd['LC'].mean()

X = forest90_pd['Fire_log'].values.reshape(-1, 1)
y = forest90_pd['Atmos_log'].values
# Theil-Sen
model = TheilSenRegressor(fit_intercept=True)
reg = model.fit(X, y)
slope[-1,0] = reg.coef_
inter = reg.intercept_
slope[-1,1] = reg.score(X, y)

plt.scatter(X, y, s=2)
plt.plot(X, X*slope[-1,0]+inter, lw=1, c='grey')

for i, b in enumerate(boundaries):
    subset = data_pd[data_pd['LC'] < b][data_pd['LC'] >= b-10]
    X = subset['Fire_log'].values.reshape(-1, 1)
    y = subset['Atmos_log'].values
    model = TheilSenRegressor(fit_intercept=True)
    reg = model.fit(X, y)
    slope[i,0] = reg.coef_
    inter = reg.intercept_
    slope[i,1] = reg.score(X, y)
    mean_veg[i] = subset['LC'].mean()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18*cm, 14*cm))
    axes.scatter(X, y, s=2)
    axes.plot(X, X*slope[-1,0]+inter, lw=1, c='grey')

# Figure displaying change in NO2:burned area ratio with forest cover bin
cm = 1/2.54
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18*cm, 14*cm))

axes.scatter(mean_veg, slope[:,0], s=5)
ax2 = axes.twinx()
ax2.plot(mean_veg, slope[:,1])
axes.set_ylabel("Slope of best fit line", size = 10)
axes.set_xlabel("Mean forest cover % in bin", size = 10)
axes.set_xlim([0, 100])
# axes.set_ylim([0.4, 0.8])
axes.set_title('Dry season')
fig.tight_layout()


### regression slope wet season
slope = np.zeros((10,2))
boundaries = np.arange(10, 100, 10)
mean_veg = np.zeros(10)

forest90_pd = data_pd[data_pd['LC'] >= 90]
mean_veg[-1] = forest90_pd['LC'].mean()

X = forest90_pd['Fire_wet'].values.reshape(-1, 1)
y = forest90_pd['Atmos_wet'].values
# Theil-Sen
model = TheilSenRegressor(fit_intercept=True)
reg = model.fit(X, y)
slope[-1,0] = reg.coef_
inter = reg.intercept_
slope[-1,1] = reg.score(X, y)

plt.scatter(X, y, s=2)
plt.plot(X, X*slope[-1,0]+slope[-1,1], lw=1, c='grey')

for i, b in enumerate(boundaries):
    subset = data_pd[data_pd['LC'] < b][data_pd['LC'] >= b-10]
    X = subset['Fire_wet'].values.reshape(-1, 1)
    y = subset['Atmos_wet'].values
    model = TheilSenRegressor(fit_intercept=True)
    reg = model.fit(X, y)
    slope[i,0] = reg.coef_
    inter = reg.intercept_
    slope[i,1] = reg.score(X, y)
    mean_veg[i] = subset['LC'].mean()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18*cm, 14*cm))
    axes.scatter(X, y, s=2)
    axes.plot(X, X*slope[-1,0]+inter, lw=1, c='grey')

# Figure displaying change in NO2:burned area ratio with forest cover bin
cm = 1/2.54
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18*cm, 14*cm))

axes.scatter(mean_veg, slope[:,0], s=5)
ax2 = axes.twinx()
ax2.plot(mean_veg, slope[:,1])
axes.set_ylabel(f"{labels[atmos_no]} : burned area", size = 10)
axes.set_xlabel("Mean forest cover % in bin", size = 10)
axes.set_xlim([0, 100])
# axes.set_ylim([0.4, 0.8])
axes.set_title('Wet season')
fig.tight_layout()