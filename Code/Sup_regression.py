# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:58:55 2023

@author: s2261807
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy.ma as ma
from sklearn.linear_model import TheilSenRegressor
import statsmodels.api as sm
from scipy.stats import spearmanr
import matplotlib as mpl

# =============================================================================
# load data
# =============================================================================

# burned area (GFED 4; 2001-2016); error in metadata - unit is fraction of cell, not %
# fn = 'R:\\gfed\\monthly_1degree_sum_2001-2016.nc'
# ds = xr.open_dataset(fn, decode_times=True)
# fire = ds['Burned area']*100 
# fire['time'] = pd.date_range('2001-01-01', '2016-12-31', freq = 'MS')
# ds.close()
fn = 'R:\\gfed\\GFED4_BAkm2_2001-2016.nc'
ds = xr.open_dataset(fn, decode_times=True)
fire = ds['Burned Area']
fire['time'] = pd.date_range('2001-01-01', '2016-12-31', freq = 'MS')
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

### isoprene
fn = 'R:/cris_isoprene/2012-2020_CrIS_monthly_Isoprene_1degree_july2017interpolated.nc'
ds = xr.open_dataset(fn, decode_times=True)
isop = ds['isoprene'][:-12,:,:]
ds.close()

### NO2
fn = 'R:/OMI_NO2/omi_no2_mm_2005_2020_masked.nc'
ds = xr.open_dataset(fn, decode_times=True)
no2 = ds['no2'][:-12,:,:]
ds.close()

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

# AOD
fn = 'R:\modis_aod\mod08_aod_masked_2001-2019.nc'
ds = xr.open_dataset(fn, decode_times=True)
aod = ds['aod'][:,0,:,:]
# # mask_value = aod[0,0,0,0]
ds.close()

# methanol
fn = 'R:/methanol/methanol_1degree_2008-2018.nc'
ds = xr.open_dataset(fn, decode_times=True)
methanol = ds['methanol'][:,:,:]
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

def get_lai_dist(atm_comp, lc, perc):
    if perc == 15:
        cover = ma.masked_less_equal(lc, perc).mask.flatten()
    else:
        cover = ma.masked_inside(lc, perc - 4.9, perc).mask.flatten()
    
    atm_comp_1d = atm_comp.values.flatten()   
    # atm_comp_1d = atm_comp.flatten()   # use for hcho
    if np.any(cover) == False:
        cover = np.zeros_like(atm_comp_1d, dtype = np.bool)
    atm_comp_dist = atm_comp_1d[cover]
    return atm_comp_dist

def get_atm_dist(atm_comp, lc, perc):
    if perc == 5:
        cover = ma.masked_less_equal(lc, perc).mask.flatten()
    else:
        cover = ma.masked_inside(lc, perc - 4.9, perc).mask.flatten()
    
    atm_comp_1d = atm_comp.values.flatten()   
    # atm_comp_1d = atm_comp.flatten()   # use for hcho
    atm_comp_dist = atm_comp_1d[cover]
    return atm_comp_dist

def get_fire_dist(atm_comp, lc, perc):
    if perc == 5:
        cover = ma.masked_less_equal(lc, perc).mask.flatten()
    if perc ==50:
        cover = ma.masked_greater(lc, 45).mask.flatten()
    else:
        cover = ma.masked_inside(lc, perc - 4.9, perc).mask.flatten()
    
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
fire1 = fire.fillna(0)/surface_area_earth *100
# =============================================================================
# crop data to region of interest
# =============================================================================

fire_crop = crop_data(fire1)
lai_crop = crop_data(lai)
broadleaf_crop = crop_data(broadleaf)
dem_crop = crop_data(dem)

hcho_crop = crop_data(hcho)
co_crop = crop_data(co)
no2_crop = crop_data(no2)
aod_crop = crop_data(aod)
isop_crop = crop_data(isop)
methanol_crop = crop_data(methanol)

# =============================================================================
# separate data into wet and dry season
# =============================================================================
fire_wet = get_month_data([2,3,4], fire_crop)
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
atmos_no = 4
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
    atmos_dry = no2_dry 
    atmos_wet = no2_wet 
elif atmos_no == 5:
    atmos_dry = co_dry 
    atmos_wet = co_wet 
elif atmos_no == 6:
    atmos_dry = hcho_dry_d 
    atmos_wet = hcho_wet_d 
else:
    print('atmos_no is out of bounds (1 to 6)')

labels = {1 : 'Isoprene', 2 : 'AOD' , 3 : 'Methanol', 4 : 'NO$_{2}$', 5 : 'CO', 6 : 'HCHO'}

units = {1 : '10$^{16}$ molecules cm$^{-2}$', 2 : 'at 0.47 $\mu$m', 3 : '[ppbv]', 4 : '[molecules cm$^{-2}$]',\
         5 : '[10$^{17}$ molecules cm$^{-2}$]', 6 : '[molecules cm$^{-2}$]'}


lcs = [broadleaf_crop]
lc_labels = ['Broadleaf Forest']

# lc_means = np.zeros((4, 10))

elev_boundary = 1000
high_elev = ma.masked_greater_equal(dem_crop, elev_boundary).mask

atmos_dry = mask_high_elev(atmos_dry, high_elev) 
atmos_wet = mask_high_elev(atmos_wet, high_elev)

for i, y in enumerate(lcs):
    lc = y
    
    # ### keeping only data for months where both datasets are available
    min_time = xr.DataArray(data = [fire_dry.time.min().values, atmos_dry.time.min().values]).max().values
    min_year = min_time.astype(str).split('-')[0]
    max_time = xr.DataArray(data = [fire_dry.time.max().values, atmos_dry.time.max().values]).min().values
    max_year = max_time.astype(str).split('-')[0]
    
    ### keeping only data for 2012-2016
    # min_time = xr.DataArray(data = [fire_dry.time.min().values, isop_dry.time.min().values]).max().values
    # min_year = min_time.astype(str).split('-')[0]
    # max_time = xr.DataArray(data = [fire_dry.time.max().values, isop_dry.time.max().values]).min().values
    # max_year = max_time.astype(str).split('-')[0]
    
    fire_slice = fire_dry.sel(time=slice(min_year, max_year))
    fire_wet_slice = fire_wet.sel(time=slice(min_year, max_year))
    atmos_dry_slice = atmos_dry.sel(time=slice(min_year, max_year))
    atmos_wet_slice = atmos_wet.sel(time=slice(min_year, max_year))
    lc_slice = lc.sel(time=slice(min_year, max_year))
    lai_slice = lai.sel(time=slice(min_year, max_year))
    #lc_int = (np.rint(lc_slice*100)).astype(int)
    lc_int = np.rint(lc_slice) #if LC B used
    
    ### masking fire occurrence
    boundary = 0.004
    no_fire = ma.masked_less_equal(fire_slice, boundary).mask
    yes_fire = ma.masked_greater(fire_slice, boundary).mask
        
    atmos_dry_fire, atmos_dry_no_fire = mask_data(atmos_dry_slice, yes_fire, no_fire)
    
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
# atmos_dry_1d = np.where(atmos_dry_1d < 10**17, atmos_dry_1d, np.nan) #for HCHO

lai_1d = reshape_lai.ravel()

lc_1d = reshape_lc.ravel()

fire_1d = fire_slice.values.ravel()
fire_wet_1d = fire_wet_slice.values.ravel()
lat_1d = reshape_lat.ravel()
lon_1d = reshape_lon.ravel()

# plt.scatter(lai_1d, atmos_dry_1d)
# plt.scatter(lc_1d, atmos_dry_1d)
# plt.scatter(fire_1d, atmos_dry_1d)

### create pandas dataframe to hold data of interest
# for_export = {'Atmos': atmos_dry_1d, 'Fire': fire_1d}
# data_export = pd.DataFrame(for_export)
# np.savetxt('C:/Users/s2261807/Documents/no2_burned_area.txt', data_export.values)

data_log = {'Atmos': atmos_dry_1d, 'LC': lc_1d, 'LAI': lai_1d, 'Fire': fire_1d, \
        'Atmos_log': np.log(atmos_dry_1d), 'Fire_log': np.log(fire_1d)}#, 'Atmos_sqrt': np.sqrt(atmos_dry_1d),\
            # 'Fire_sqrt': np.sqrt(fire_1d), 'Lat' : lat_1d, 'Lon' : lon_1d} #, 'expFire': np.exp(fire_1d)}

data = {'Atmos': atmos_dry_1d, 'Atmos_wet': atmos_wet_1d, 'LC': lc_1d, 'LAI': lai_1d,\
        'Fire': fire_1d, 'Fire_wet': fire_wet_1d, 'Lat' : lat_1d, 'Lon' : lon_1d}
    

data_pd = pd.DataFrame(data).replace(-np.Inf, np.nan).dropna()
data_log_pd = pd.DataFrame(data_log).replace(-np.Inf, np.nan).dropna()



print(spearmanr(data_pd['Atmos'], data_pd['LC']))
print(spearmanr(data_pd['Atmos'], data_pd['LAI']))
print(spearmanr(data_pd['Atmos'], data_pd['Fire']))
print(spearmanr(data_log_pd['Atmos_log'], data_log_pd['Fire_log']))

### OLS
# Xi = data_pd['LC']
# Xi = data_pd['LAI']
# Xi = data_pd['Fire']
Xi = data_log_pd['Fire_log']
# Xi = data_pd[['LC', 'Fire']]
# Xi = data_pd[['Fire', 'LAI']]
# y = data_pd['Atmos']
y = data_log_pd['Atmos_log']
X = sm.add_constant(Xi)
mod = sm.OLS(y, X)
res = mod.fit()

print(res.summary())


### TS
reg = TheilSenRegressor().fit(X, y)
score = reg.score(X, y)
print(score)
# =============================================================================
# NO2 figure - points coloured based on lai and lc
# =============================================================================
cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
axes = axes.ravel()
cmap = plt.cm.get_cmap('viridis')
# cmap = plt.cm.get_cmap('coolwarm')
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
# define the bins and normalize
bounds2 = np.linspace(0, 100, 11)
norm2 = mpl.colors.BoundaryNorm(bounds2, cmap.N)
bounds1 = np.linspace(0, 6, 7)
norm1 = mpl.colors.BoundaryNorm(bounds1, cmap.N)
bounds3 = np.linspace(-25, -5, 11)
norm3 = mpl.colors.BoundaryNorm(bounds3, cmap.N)
bounds4 = np.linspace(-70, -50, 11)
norm4 = mpl.colors.BoundaryNorm(bounds4, cmap.N)

a = axes[0].scatter(data_pd['Fire'], data_pd['Atmos']/10**16, s=2, c=data_pd['LC'], cmap=cmap, norm=norm2)
fig.colorbar(a,ax=axes[0],orientation='vertical',label='Forest Cover')
fig.text(0.02, 0.95, '(a)', fontsize = 10)

b = axes[1].scatter(data_pd['Fire'], data_pd['Atmos']/10**16, s=2, c=data_pd['LAI']/10, cmap=cmap, norm=norm1)
fig.colorbar(b,ax=axes[1],orientation='vertical',label='LAI')
fig.text(0.51, 0.95, '(b)', fontsize = 10)

c = axes[2].scatter(data_pd['Fire'], data_pd['Atmos']/10**16, s=2, c=data_pd['Lat'], cmap=cmap, norm=norm3)
fig.colorbar(c,ax=axes[2],orientation='vertical',label='Latitude')
fig.text(0.02, 0.46, '(c)', fontsize = 10)

d = axes[3].scatter(data_pd['Fire'], data_pd['Atmos']/10**16, s=2, c=data_pd['Lon'], cmap=cmap, norm=norm4)
fig.colorbar(d,ax=axes[3],orientation='vertical',label='Longitude')
fig.text(0.51, 0.46, '(d)', fontsize = 10)
for i in range(4):
    axes[i].set_ylabel('NO$_{2}$ [10$^{16}$ mol cm$^{-2}$]', size = 10)
    axes[i].set_xlabel('Burned area [% grid cell area]', size = 10)
    axes[i].set_xlim([0, 2])
    axes[i].set_ylim([0,1.5])
    axes[i].set_yticks(np.arange(0, 1.6, 0.5))

fig.tight_layout()

# fig.savefig('C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/no2_dry_season_coloured_scatter.pdf')
# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/no2_dry_season_coloured_scatter.png', dpi = 300)

cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
axes = axes.ravel()

a = axes[0].scatter(data_pd['Fire_log'], data_pd['Atmos_log'], s=2, c=data_pd['LAI']/10, cmap='YlGnBu')
fig.colorbar(a,ax=axes[0],orientation='vertical',label='LAI')

b = axes[1].scatter(data_pd['Fire_log'], data_pd['Atmos_log'], s=2, c=data_pd['LC'], cmap='YlGnBu')
fig.colorbar(b,ax=axes[1],orientation='vertical',label='Forest Cover')
c = axes[2].scatter(data_pd['Fire_log'], data_pd['Atmos_log'], s=2, c=data_pd['Lat'], cmap='YlGnBu')
fig.colorbar(c,ax=axes[2],orientation='vertical',label='Latitude')
d = axes[3].scatter(data_pd['Fire_log'], data_pd['Atmos_log'], s=2, c=data_pd['Lon'], cmap='YlGnBu')
fig.colorbar(d,ax=axes[3],orientation='vertical',label='Longitude')
for i in range(4):
    axes[i].set_ylabel(f'Nat log of {labels[atmos_no]} {units[atmos_no]}', size = 10)
    axes[i].set_xlabel('Nat log of burned area \n[% grid cell area]', size = 10)
fig.tight_layout()

# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/log{labels[atmos_no]}_dry_season_coloured_scatter.png', dpi = 300)
# 

### focusing on specific forest covers
forest75_pd = data_pd[data_pd['LC'] > 75]
forest50_pd = data_pd[data_pd['LC'] <= 75][data_pd['LC'] > 50]
forest25_pd = data_pd[data_pd['LC'] <= 50][data_pd['LC'] > 25]
forest0_pd = data_pd[data_pd['LC'] <= 25]

cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
axes = axes.ravel()

a = axes[0].scatter(forest75_pd['Fire'], forest75_pd['Atmos'], s=2, c=forest75_pd['LC'], cmap='YlGnBu')
fig.colorbar(a,ax=axes[0],orientation='vertical',label='Forest Cover')
b = axes[1].scatter(forest50_pd['Fire'], forest50_pd['Atmos'], s=2, c=forest50_pd['LC'], cmap='YlGnBu')
fig.colorbar(b,ax=axes[1],orientation='vertical',label='Forest Cover')
c = axes[2].scatter(forest25_pd['Fire'], forest25_pd['Atmos'], s=2, c=forest25_pd['LC'], cmap='YlGnBu')
fig.colorbar(c,ax=axes[2],orientation='vertical',label='Forest Cover')
d = axes[3].scatter(forest0_pd['Fire'], forest0_pd['Atmos'], s=2, c=forest0_pd['LC'], cmap='YlGnBu')
fig.colorbar(d,ax=axes[3],orientation='vertical',label='Forest Cover')
for i in range(4):
    axes[i].set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 10)
    axes[i].set_xlabel('Burned area [% grid cell area]', size = 10)
    axes[i].set_xlim([0,0.5])
    axes[i].set_ylim([0,1.6 * 10**16])
# axes[0].set_xlim([0,0.02])
fig.tight_layout()

# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_dry_season_forest_dependency_corrGFED4.png', dpi = 300)

cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
axes = axes.ravel()

a = axes[0].scatter(forest75_pd['Fire_log'], forest75_pd['Atmos_log'], s=2, c=forest75_pd['LC'], cmap='YlGnBu')
fig.colorbar(a,ax=axes[0],orientation='vertical',label='Forest Cover')
b = axes[1].scatter(forest50_pd['Fire_log'], forest50_pd['Atmos_log'], s=2, c=forest50_pd['LC'], cmap='YlGnBu')
fig.colorbar(b,ax=axes[1],orientation='vertical',label='Forest Cover')
c = axes[2].scatter(forest25_pd['Fire_log'], forest25_pd['Atmos_log'], s=2, c=forest25_pd['LC'], cmap='YlGnBu')
fig.colorbar(c,ax=axes[2],orientation='vertical',label='Forest Cover')
d = axes[3].scatter(forest0_pd['Fire_log'], forest0_pd['Atmos_log'], s=2, c=forest0_pd['LC'], cmap='YlGnBu')
fig.colorbar(d,ax=axes[3],orientation='vertical',label='Forest Cover')
for i in range(4):
    axes[i].set_ylabel(f'Log {labels[atmos_no]} {units[atmos_no]}', size = 10)
    axes[i].set_xlabel('Log Burned area [% grid cell area]', size = 10)
    axes[i].set_xlim([-10, -2.5])
    axes[i].set_ylim([33, 38])
fig.tight_layout()

# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/log{labels[atmos_no]}_dry_season_forest_dependency.png', dpi = 300)
# cm = 1/2.54
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
# axes = axes.ravel()

# a = axes[0].scatter(data_pd['Fire_wet'], data_pd['Atmos'], s=2, c=data_pd['LAI']/10, cmap='YlGnBu')
# fig.colorbar(a,ax=axes[0],orientation='vertical',label='LAI')

# b = axes[1].scatter(data_pd['Fire_wet'], data_pd['Atmos'], s=2, c=data_pd['LC'], cmap='YlGnBu')
# fig.colorbar(b,ax=axes[1],orientation='vertical',label='Forest Cover')
# c = axes[2].scatter(data_pd['Fire_wet'], data_pd['Atmos'], s=2, c=data_pd['Lat'], cmap='YlGnBu')
# fig.colorbar(c,ax=axes[2],orientation='vertical',label='Latitude')
# d = axes[3].scatter(data_pd['Fire_wet'], data_pd['Atmos'], s=2, c=data_pd['Lon'], cmap='YlGnBu')
# fig.colorbar(d,ax=axes[3],orientation='vertical',label='Longitude')
# for i in range(4):
#     axes[i].set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 10)
#     axes[i].set_xlabel('Burned area [% grid cell area]', size = 10)
    
# fig.tight_layout()

# # fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_dry_season_coloured_scatter.png', dpi = 300)

# cm = 1/2.54
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
# axes = axes.ravel()

# a = axes[0].scatter(data_pd['Fire_wet_log'], data_pd['Atmos_log'], s=2, c=data_pd['LAI']/10, cmap='YlGnBu')
# fig.colorbar(a,ax=axes[0],orientation='vertical',label='LAI')

# b = axes[1].scatter(data_pd['Fire_wet_log'], data_pd['Atmos_log'], s=2, c=data_pd['LC'], cmap='YlGnBu')
# fig.colorbar(b,ax=axes[1],orientation='vertical',label='Forest Cover')
# c = axes[2].scatter(data_pd['Fire_wet_log'], data_pd['Atmos_log'], s=2, c=data_pd['Lat'], cmap='YlGnBu')
# fig.colorbar(c,ax=axes[2],orientation='vertical',label='Latitude')
# d = axes[3].scatter(data_pd['Fire_wet_log'], data_pd['Atmos_log'], s=2, c=data_pd['Lon'], cmap='YlGnBu')
# fig.colorbar(d,ax=axes[3],orientation='vertical',label='Longitude')
# for i in range(4):
#     axes[i].set_ylabel(f'Nat log of {labels[atmos_no]} {units[atmos_no]}', size = 10)
#     axes[i].set_xlabel('Nat log of burned area \n[% grid cell area]', size = 10)
    
# fig.tight_layout()

# # fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/log{labels[atmos_no]}_dry_season_coloured_scatter.png', dpi = 300)
# 
# =============================================================================
# Isoprene colored by burned area
# =============================================================================
cm = 1/2.54
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18*cm, 14*cm))
axes = axes.ravel()

a = axes[0].scatter(data_pd['LC'], data_pd['Atmos'], s=2, c=data_pd['LAI']/10, cmap='YlGnBu')
fig.colorbar(a,ax=axes[0],orientation='vertical',label='LAI')

b = axes[1].scatter(data_pd['LC'], data_pd['Atmos'], s=2, c=data_pd['Fire'], cmap='YlGnBu')
fig.colorbar(b,ax=axes[1],orientation='vertical',label='Burned area')
c = axes[2].scatter(data_pd['LC'], data_pd['Atmos'], s=2, c=data_pd['Lat'], cmap='YlGnBu')
fig.colorbar(c,ax=axes[2],orientation='vertical',label='Latitude')
d = axes[3].scatter(data_pd['LC'], data_pd['Atmos'], s=2, c=data_pd['Lon'], cmap='YlGnBu')
fig.colorbar(d,ax=axes[3],orientation='vertical',label='Longitude')
for i in range(4):
    axes[i].set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 10)
    axes[i].set_xlabel('Forest cover', size = 10)
    
fig.tight_layout()

# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_dry_season_coloured_scatter.png', dpi = 300)

# =============================================================================
# binning - alternative method
# =============================================================================
Xi_all = data_log_pd['Fire_log']
# Xi = data_pd[['LC', 'Fire']]
y_all = data_log_pd['Atmos_log']

X_all = sm.add_constant(Xi_all)

reg = TheilSenRegressor().fit(X_all, y_all)
# reg = OLS().fit(X, y)

score = reg.score(X_all, y_all)
print(score)
y_pred_all = reg.predict(X_all)

import scipy.stats
sorted_fire = data_pd.sort_values('Fire')

bin_edges = np.arange(0, 0.5, 0.025)
bin_edges = np.append(bin_edges, 0.5)

test_atm = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], bins= bin_edges)[0]
test_fire = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], bins= bin_edges)[0]
test_error = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], statistic = 'std', bins= bin_edges)[0] \
    / np.sqrt(scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], statistic = 'count', bins= bin_edges)[0])
fire_error = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], statistic = 'std', bins= bin_edges)[0] \
    / np.sqrt(scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], statistic = 'count', bins= bin_edges)[0])



err_max = test_atm + test_error
err_min = test_atm - test_error

ferr_max = test_fire + fire_error
ferr_min = test_fire - fire_error

Xi = np.log(test_fire)
# Xi = data_pd[['LC', 'Fire']]
y = np.log(test_atm)
X = sm.add_constant(Xi)

### weighted least squares

mod_weighted = sm.WLS(y, X, weights = 1/test_error, missing = 'drop')
reg_weighted = mod_weighted.fit()
print(reg_weighted.summary())
y_pred = reg_weighted.predict(X)

log_min = np.log(err_min)
log_max = np.log(err_max)
log_errors  =np.zeros((2, len(log_min)))
log_errors[0,:] = y - log_min 
log_errors[1,:] = log_max - y

flog_min = np.log(ferr_min)
flog_max = np.log(ferr_max)
flog_errors  =np.zeros((2, len(flog_min)))
flog_errors[0,:] = Xi - flog_min 
flog_errors[1,:] = flog_max - Xi

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
# axes.errorbar(Xi, y, log_errors, flog_errors, linestyle = '',\
#                   marker = 'o', markersize = 2, capsize = 4, c = 'grey')
# # axes.scatter(Xi, y, marker = 'o', facecolors = 'none', edgecolor = 'grey')
# axes.plot(Xi, y_pred, label = 'Predicted', c = 'navy')
# axes.set_ylabel(f'Nat log of mean {labels[atmos_no]} {units[atmos_no]}', size = 16)
# axes.set_xlabel('Nat log of mean burned area [% grid cell area]', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'
# fig.text(0.3, 0.7, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}', size = 12, c = 'navy') # \nR$^{2}$ = 0.98
# # axes.set_ylim(ymin = 34.5, ymax = 36)



resid_WLS = np.zeros(20)
for i in range(20):
    resid_WLS[i] = y[i] - (reg_weighted.params[0] + Xi[i]*reg_weighted.params[1])


fontsize = 8
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12*cm, 18*cm))
axes = axes.ravel()
axes[0].scatter(Xi_all, y_all, s = 2, c='grey', cmap='YlGnBu')#c = data_pd['LC'])
axes[0].plot(Xi_all, y_pred_all, c ='navy')
axes[0].set_ylabel('log$_{e}$ ' + f'of {labels[atmos_no]}', size = fontsize)
axes[0].set_xlabel('log$_{e}$ of burned area', size = fontsize)
axes[0].tick_params(labelsize=fontsize)

axes[1].errorbar(Xi, y,  log_errors, linestyle = '',\
                  marker = 'o', markersize = 2, capsize = 4, c = 'grey')
# axes[1].errorbar(Xi, y,  test_error, linestyle = '',\
#                   marker = 'o', markersize = 2, capsize = 4, c = 'grey')

axes[1].plot(Xi, reg_weighted.params[0] + Xi*reg_weighted.params[1], c ='navy')
axes[1].set_ylabel('log$_{e}$ ' + f'of {labels[atmos_no]}', size = fontsize)
axes[1].set_xlabel('log$_{e}$ of burned area', size = fontsize)
axes[1].set_yticks(np.arange(35, 36.5, 0.5))
axes[1].tick_params(labelsize=fontsize)

axes[2].hist(resid_WLS, color = 'grey')
axes[2].set_ylabel('Residual count', size = fontsize)
axes[2].set_xlabel('log$_{e}$(NO$_{2}$) residuals for the WLS regression', size = fontsize)
axes[2].tick_params(labelsize=fontsize)

# axes.tick_params(axis='both', which='major', labelsize=14)
# fig.text(0.23, 0.89, f'y = {reg.coef_[1]:.2e}*x + {reg.coef_[0]:.2e}  \nR$^{2}$ = {score:.2f}', size = 12, c = 'navy')
# fig.text(0.23, 0.57, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = {reg_weighted.rsquared:.2f}', size = 12, c = 'navy')

fig.text(0.2, 0.95, r'log$_{e}$(NO$_{2}$) = 1.98 $\times$ 10$^{-1}$ $\times$ log$_{e}$(BA) $\plus$ 18', size = fontsize, c = 'navy') 
fig.text(0.2, 0.91, f'R$^{2}$ = {score:.2f}', size = fontsize, c = 'navy')
fig.text(0.2, 0.62, r'log$_{e}$(NO$_{2}$) = 2.61 $\times$ 10$^{-1}$ $\times$ log$_{e}$(BA) $\plus$ 36.1', size = fontsize, c = 'navy') 
fig.text(0.2, 0.58, f'R$^{2}$ = {reg_weighted.rsquared:.2f}', size = fontsize, c = 'navy')


fig.text(0.03, 0.95, '(a)', size = fontsize)
fig.text(0.03, 0.65, '(b)', size = fontsize)
fig.text(0.03, 0.33, '(c)', size = fontsize)

fig.tight_layout()

# 1 column?
fontsize = 7
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.3*cm, 13*cm))
axes = axes.ravel()
axes[0].scatter(Xi_all, y_all, s = 2, c='grey', cmap='YlGnBu')#c = data_pd['LC'])
axes[0].plot(Xi_all, y_pred_all, c ='navy')
axes[0].set_ylabel('log$_{e}$ ' + f'of {labels[atmos_no]}', size = fontsize)
axes[0].set_xlabel('log$_{e}$ of burned area', size = fontsize)
axes[0].tick_params(labelsize=fontsize)

axes[1].errorbar(Xi, y,  log_errors, linestyle = '',\
                  marker = 'o', markersize = 2, capsize = 4, c = 'grey')
# axes[1].errorbar(Xi, y,  test_error, linestyle = '',\
#                   marker = 'o', markersize = 2, capsize = 4, c = 'grey')

axes[1].plot(Xi, reg_weighted.params[0] + Xi*reg_weighted.params[1], c ='navy')
axes[1].set_ylabel('log$_{e}$ ' + f'of {labels[atmos_no]}', size = fontsize)
axes[1].set_xlabel('log$_{e}$ of burned area', size = fontsize)
axes[1].set_yticks(np.arange(35, 36.5, 0.5))
axes[1].tick_params(labelsize=fontsize)

axes[2].hist(resid_WLS, color = 'grey')
axes[2].set_ylabel('Residual count', size = fontsize)
axes[2].set_xlabel('log$_{e}$(NO$_{2}$) residuals for the WLS regression', size = fontsize)
axes[2].tick_params(labelsize=fontsize)

# axes.tick_params(axis='both', which='major', labelsize=14)
# fig.text(0.23, 0.89, f'y = {reg.coef_[1]:.2e}*x + {reg.coef_[0]:.2e}  \nR$^{2}$ = {score:.2f}', size = 12, c = 'navy')
# fig.text(0.23, 0.57, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = {reg_weighted.rsquared:.2f}', size = 12, c = 'navy')

fig.text(0.21, 0.95, r'log$_{e}$(NO$_{2}$) = 1.98 $\times$ 10$^{-1}$ $\times$ log$_{e}$(BA) $\plus$ 18', size = fontsize, c = 'navy') 
fig.text(0.21, 0.91, f'R$^{2}$ = {score:.2f}', size = fontsize, c = 'navy')
fig.text(0.21, 0.62, r'log$_{e}$(NO$_{2}$) = 2.61 $\times$ 10$^{-1}$ $\times$ log$_{e}$(BA) $\plus$ 36.1', size = fontsize, c = 'navy') 
fig.text(0.21, 0.58, f'R$^{2}$ = {reg_weighted.rsquared:.2f}', size = fontsize, c = 'navy')


fig.text(0.03, 0.95, '(a)', size = fontsize)
fig.text(0.03, 0.65, '(b)', size = fontsize)
fig.text(0.03, 0.33, '(c)', size = fontsize)

fig.tight_layout()


# save figure
# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_dry_season_regression_summary_corrGFED4.png', dpi = 300)
# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_dry_season_regression_summary.pdf')
# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_dry_season_regression_summary_1col.pdf')

# bin_edges = np.arange(0, 500/10000, 10/10000)
# bin_edges = np.append(bin_edges, 0.5)

# test_atm = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], bins= bin_edges)[0]
# test_fire = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], bins= bin_edges)[0]
# test_error = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], statistic = 'std', bins= bin_edges)[0] \
#     / np.sqrt(scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], statistic = 'count', bins= bin_edges)[0])

# Xi = np.log(test_fire)
# # Xi = data_pd[['LC', 'Fire']]
# y = np.log(test_atm)
# X = sm.add_constant(Xi)

# ### weighted least squares

# mod_weighted = sm.WLS(y, X, weights = 1/test_error, missing = 'drop')
# reg_weighted = mod_weighted.fit()
# print(reg_weighted.summary())
# y_pred = reg_weighted.predict(X)
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
# axes.scatter(Xi, y, marker = 'o', facecolors = 'none', edgecolor = 'grey')
# axes.plot(Xi, y_pred, label = 'Predicted', c = 'navy')
# axes.set_ylabel(f'Nat log of mean {labels[atmos_no]} {units[atmos_no]}', size = 16)
# axes.set_xlabel('Nat log of mean burned area [% grid cell area]', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'
# fig.text(0.3, 0.7, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}', size = 12, c = 'navy') # \nR$^{2}$ = 0.98

# =============================================================================
# Data separated by forest cover
# =============================================================================
database = other_pd
Xi_all = database['Fire_log']
# Xi = data_pd[['LC', 'Fire']]
y_all = database['Atmos_log']

X_all = sm.add_constant(Xi_all)

reg = TheilSenRegressor().fit(X_all, y_all)
# reg = OLS().fit(X, y)

score = reg.score(X_all, y_all)
print(score)
y_pred_all = reg.predict(X_all)

import scipy.stats
sorted_fire = database.sort_values('Fire')

bin_edges = np.arange(0, 500/10000, 25/10000)
bin_edges = np.append(bin_edges, 0.5)

test_atm = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], bins= bin_edges)[0]
test_fire = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], bins= bin_edges)[0]
test_error = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], statistic = 'std', bins= bin_edges)[0] \
    / np.sqrt(scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Atmos'], statistic = 'count', bins= bin_edges)[0])
fire_error = scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], statistic = 'std', bins= bin_edges)[0] \
    / np.sqrt(scipy.stats.binned_statistic(sorted_fire['Fire'], sorted_fire['Fire'], statistic = 'count', bins= bin_edges)[0])



err_max = test_atm + test_error
err_min = test_atm - test_error

ferr_max = test_fire + fire_error
ferr_min = test_fire - fire_error

Xi = np.log(test_fire)
# Xi = data_pd[['LC', 'Fire']]
y = np.log(test_atm)
X = sm.add_constant(Xi)

### weighted least squares

mod_weighted = sm.WLS(y, X, weights = 1/test_error, missing = 'drop')
reg_weighted = mod_weighted.fit()
print(reg_weighted.summary())
y_pred = reg_weighted.predict(X)

log_min = np.log(err_min)
log_max = np.log(err_max)
log_errors  =np.zeros((2, len(log_min)))
log_errors[0,:] = y - log_min 
log_errors[1,:] = log_max - y

flog_min = np.log(ferr_min)
flog_max = np.log(ferr_max)
flog_errors  =np.zeros((2, len(flog_min)))
flog_errors[0,:] = Xi - flog_min 
flog_errors[1,:] = flog_max - Xi

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
# axes.errorbar(Xi, y, log_errors, flog_errors, linestyle = '',\
#                  marker = 'o', markersize = 2, capsize = 4, c = 'grey')
# # axes.scatter(Xi, y, marker = 'o', facecolors = 'none', edgecolor = 'grey')
# axes.plot(Xi, y_pred, label = 'Predicted', c = 'navy')
# axes.set_ylabel(f'Nat log of mean {labels[atmos_no]} {units[atmos_no]}', size = 16)
# axes.set_xlabel('Nat log of mean burned area [% grid cell area]', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'
# fig.text(0.3, 0.7, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}', size = 12, c = 'navy') # \nR$^{2}$ = 0.98
# # axes.set_ylim(ymin = 34.5, ymax = 36)



resid_WLS = np.zeros(20)
for i in range(20):
    resid_WLS[i] = y[i] - (reg_weighted.params[0] + Xi[i]*reg_weighted.params[1])



cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12*cm, 18*cm))
axes = axes.ravel()
axes[0].scatter(Xi_all, y_all, s = 2, c=other_pd['LC'], cmap='YlGnBu')#c = 'grey')
axes[0].plot(Xi_all, y_pred_all, c ='navy')
axes[0].set_ylabel(f'Natural log of {labels[atmos_no]} \n{units[atmos_no]}', size = 10)
axes[0].set_xlabel('Natural log of burned area', size = 10)

axes[1].errorbar(Xi, y,  log_errors, linestyle = '',\
                 marker = 'o', markersize = 2, capsize = 4, c = 'grey')
axes[1].plot(Xi, reg_weighted.params[0] + Xi*reg_weighted.params[1], c ='navy')
axes[1].set_ylabel(f'Natural log of {labels[atmos_no]} \n{units[atmos_no]}', size = 10)
axes[1].set_xlabel('Natural log of burned area', size = 10)

axes[2].hist(resid_WLS, color = 'grey')
axes[2].set_ylabel('Residual count', size = 10)
axes[2].set_xlabel('log(NO2) residuals for the WLS regression', size = 10)


# axes.tick_params(axis='both', which='major', labelsize=14)
# fig.text(0.23, 0.89, f'y = {reg.coef_[1]:.2e}*x + {reg.coef_[0]:.2e}  \nR$^{2}$ = {score:.2f}', size = 12, c = 'navy')
# fig.text(0.23, 0.57, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = {reg_weighted.rsquared:.2f}', size = 12, c = 'navy')

fig.text(0.23, 0.95, r'$log_{e}(NO_2) = 1.82 \times 10^{-1} \times log_{e}(BA) \plus 18$', size = 11, c = 'navy') 
fig.text(0.23, 0.91, f'R$^{2}$ = {score:.2f}', size = 11, c = 'navy')
fig.text(0.23, 0.62, r'$log_{e}(NO_2) = 2.59 \times 10^{-1} \times log_{e}(BA) \plus 36.7$', size = 11, c = 'navy') 
fig.text(0.23, 0.58, f'R$^{2}$ = {reg_weighted.rsquared:.2f}', size = 11, c = 'navy')


fig.text(0.03, 0.95, '(a)', size = 14)
fig.text(0.03, 0.65, '(b)', size = 14)
fig.text(0.03, 0.33, '(c)', size = 14)

fig.tight_layout()
# 
# =============================================================================
# Isoprene regression figure
# =============================================================================
Xi_all = data_pd['LC']
# Xi = data_pd[['LC', 'Fire']]
y_all = data_pd['Atmos']

X_all = sm.add_constant(Xi_all)

reg = TheilSenRegressor().fit(X_all, y_all)
# reg = OLS().fit(X, y)

score = reg.score(X_all, y_all)
print(score)
y_pred_all = reg.predict(X_all)

# import scipy.stats
sorted_lc = data_pd.sort_values('LC')

bin_edges = np.arange(0, 500/10000, 25/10000)
bin_edges = np.append(bin_edges, 0.5)

test_atm = scipy.stats.binned_statistic(sorted_lc['LC'], sorted_lc['Atmos'], bins= 20)[0]
test_lc = scipy.stats.binned_statistic(sorted_lc['LC'], sorted_lc['LC'], bins= 20)[0]
test_error = scipy.stats.binned_statistic(sorted_lc['LC'], sorted_lc['Atmos'], statistic = 'std', bins= 20)[0] \
    / np.sqrt(scipy.stats.binned_statistic(sorted_lc['LC'], sorted_lc['Atmos'], statistic = 'count', bins= 20)[0])
lc_error = scipy.stats.binned_statistic(sorted_lc['LC'], sorted_lc['LC'], statistic = 'std', bins= 20)[0] \
    / np.sqrt(scipy.stats.binned_statistic(sorted_lc['LC'], sorted_lc['LC'], statistic = 'count', bins= 20)[0])


Xi = test_lc
y = test_atm
X = sm.add_constant(Xi)

### weighted least squares
mod_weighted = sm.WLS(y, X, weights = 1/test_error, missing = 'drop')
reg_weighted = mod_weighted.fit()

resid_WLS = np.zeros(20)
for i in range(20):
    resid_WLS[i] = test_atm[i] - (reg_weighted.params[0] + Xi[i]*reg_weighted.params[1])


fontsize = 9
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10*cm, 14*cm))
axes = axes.ravel()
axes[0].scatter(Xi_all, y_all/10**16, s = 2, c = 'grey')
axes[0].plot(Xi_all, y_pred_all/10**16, c ='navy')
axes[0].set_ylabel(f'{labels[atmos_no]} \n{units[atmos_no]}', size = fontsize)
axes[0].set_xlabel('Broadleaf Forest Cover %', size = fontsize)
axes[0].tick_params(labelsize=fontsize)

axes[1].errorbar(Xi, y/10**16,  test_error/10**16, linestyle = '',\
                 marker = 'o', markersize = 2, capsize = 4, c = 'grey')
axes[1].plot(Xi, reg_weighted.params[0]/10**16 + Xi*reg_weighted.params[1]/10**16, c ='navy')
axes[1].set_ylabel(f'{labels[atmos_no]} \n{units[atmos_no]}', size = fontsize)
axes[1].set_xlabel('Broadleaf Forest Cover %', size = fontsize)
axes[1].set_yticks(np.arange(0.5, 1.6, 0.5))
axes[1].tick_params(labelsize=fontsize)

axes[2].hist(resid_WLS/10**16, color = 'grey')
axes[2].set_ylabel('Residual count', size = fontsize)
axes[2].set_xlabel(f'Isoprene residuals for the WLS regression \n{units[atmos_no]}', size = fontsize)
axes[2].tick_params(labelsize=fontsize)

# axes.tick_params(axis='both', which='major', labelsize=14)
# fig.text(0.23, 0.89, f'y = {reg.coef_[1]:.2e}*x + {reg.coef_[0]:.2e}  \nR$^{2}$ = {score:.2f}', size = 12, c = 'navy')
# fig.text(0.23, 0.57, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = {reg_weighted.rsquared:.2f}', size = 12, c = 'navy')

fig.text(0.23, 0.94, r'Isop = 1.12 $\times$ 10$^{14}$ $\times$ BFC $\plus$ 1.26 $\times$ 10$^{15}$', size = fontsize, c = 'navy') 
fig.text(0.23, 0.9, f'R$^{2}$ = {score:.2f}', size = fontsize, c = 'navy')
fig.text(0.23, 0.63, r'Isop = 1.08 $\times$ 10$^{14}$ $\times$ BFC $\plus$ 2.66 $\times$ 10$^{15}$', size = fontsize, c = 'navy') 
fig.text(0.23, 0.58, f'R$^{2}$ = {reg_weighted.rsquared:.2f}', size = fontsize, c = 'navy')



fig.text(0.03, 0.95, '(a)', size = fontsize)
fig.text(0.03, 0.65, '(b)', size = fontsize)
fig.text(0.03, 0.33, '(c)', size = fontsize)

fig.tight_layout()


fontsize = 7
cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.3*cm, 13*cm))
axes = axes.ravel()
axes[0].scatter(Xi_all, y_all/10**16, s = 2, c = 'grey')
axes[0].plot(Xi_all, y_pred_all/10**16, c ='navy')
axes[0].set_ylabel(f'{labels[atmos_no]} \n{units[atmos_no]}', size = fontsize)
axes[0].set_xlabel('Broadleaf Forest Cover %', size = fontsize)
axes[0].tick_params(labelsize=fontsize)

axes[1].errorbar(Xi, y/10**16,  test_error/10**16, linestyle = '',\
                 marker = 'o', markersize = 2, capsize = 4, c = 'grey')
axes[1].plot(Xi, reg_weighted.params[0]/10**16 + Xi*reg_weighted.params[1]/10**16, c ='navy')
axes[1].set_ylabel(f'{labels[atmos_no]} \n{units[atmos_no]}', size = fontsize)
axes[1].set_xlabel('Broadleaf Forest Cover %', size = fontsize)
axes[1].set_yticks(np.arange(0.5, 1.6, 0.5))
axes[1].tick_params(labelsize=fontsize)

axes[2].hist(resid_WLS/10**16, color = 'grey')
axes[2].set_ylabel('Residual count', size = fontsize)
axes[2].set_xlabel(f'Isoprene residuals for the WLS regression \n{units[atmos_no]}', size = fontsize)
axes[2].tick_params(labelsize=fontsize)

# axes.tick_params(axis='both', which='major', labelsize=14)
# fig.text(0.23, 0.89, f'y = {reg.coef_[1]:.2e}*x + {reg.coef_[0]:.2e}  \nR$^{2}$ = {score:.2f}', size = 12, c = 'navy')
# fig.text(0.23, 0.57, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = {reg_weighted.rsquared:.2f}', size = 12, c = 'navy')

fig.text(0.23, 0.94, r'Isop = 1.12 $\times$ 10$^{14}$ $\times$ BFC $\plus$ 1.26 $\times$ 10$^{15}$', size = fontsize, c = 'navy') 
fig.text(0.23, 0.9, f'R$^{2}$ = {score:.2f}', size = fontsize, c = 'navy')
fig.text(0.23, 0.63, r'Isop = 1.08 $\times$ 10$^{14}$ $\times$ BFC $\plus$ 2.66 $\times$ 10$^{15}$', size = fontsize, c = 'navy') 
fig.text(0.23, 0.58, f'R$^{2}$ = {reg_weighted.rsquared:.2f}', size = fontsize, c = 'navy')



fig.text(0.03, 0.95, '(a)', size = fontsize)
fig.text(0.03, 0.65, '(b)', size = fontsize)
fig.text(0.03, 0.33, '(c)', size = fontsize)

fig.tight_layout()
# save figure
# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_dry_season_regression_summary_formatted.png', dpi = 300)
# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_dry_season_regression_summary_formatted.pdf')
# fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_dry_season_regression_summary_formatted_1col.pdf')
# 
###

# dry season all
data3 = []
data3_len = []
data4 = []
data4_len = []

for p in range(5, 101, 5): # or 51 for urban
    distribution_data = get_atm_dist(atmos_wet_slice, reshape_lc, p)
    distribution_data = distribution_data[~np.isnan(distribution_data)]
    data3.append(distribution_data)
    data3_len.append(len(distribution_data))
    
for p in range(5, 101, 5): # or 51 for urban
    distribution_data = get_atm_dist(fire_slice, reshape_lc, p)
    distribution_data = distribution_data[~np.isnan(distribution_data)]
    data4.append(distribution_data)
    data4_len.append(len(distribution_data))
    
def calc_statistic2(data, stat = 'median'):
    data_means = np.zeros(20)
    if stat == 'mean':
        for i,a in enumerate(data):
            data_means[i]=a.mean()
    elif stat == 'median':
        for i,a in enumerate(data):
            data_means[i]=np.median(a)  
    elif stat == 'std':
        for i,a in enumerate(data):
            data_means[i]=np.std(a)          
    else:
        print('invalid stat')
    return data_means
        
data3_means = calc_statistic2(data3, stat = 'mean')
data4_means = calc_statistic2(data4, stat= 'mean')
data3_std = calc_statistic2(data3, stat = 'std')
data4_std = calc_statistic2(data4, stat= 'std')
data3_err = calc_statistic2(data3, stat = 'std')/np.sqrt(data3_len)
data4_err = calc_statistic2(data4, stat= 'std')/np.sqrt(data4_len)
lcs = np.arange(2.5, 98, 5)

data = {'Atmos': data3_means, 'LC': lcs, 'LC_log': np.log(lcs), 'Fire': data4_means} #'LAI': lai_1d, } #, 'expFire': np.exp(fire_1d)}
data_pd = pd.DataFrame(data)
Xi = data_pd['LC']
y = data_pd['Atmos']
X = sm.add_constant(Xi)

### weighted least squares
mod_weighted = sm.WLS(y, X, weights = 1/data3_err, missing = 'drop')
reg_weighted = mod_weighted.fit()

resid_WLS = np.zeros(20)
for i in range(20):
    resid_WLS[i] = data3_means[i] - (reg_weighted.params[0] + Xi[i]*reg_weighted.params[1])



cm = 1/2.54
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12*cm, 18*cm))
axes = axes.ravel()
axes[0].scatter(Xi_all, y_all/10**16, marker = 'o', facecolors = 'none', edgecolors = 'grey')
axes[0].plot(Xi_all, y_pred_all/10**16, c ='navy')
axes[0].set_ylabel(f'{labels[atmos_no]} \n{units[atmos_no]}', size = 10)
axes[0].set_xlabel('Broadleaf Forest Cover %', size = 10)

axes[1].errorbar(Xi, y/10**16,  data3_err/10**16, linestyle = '',\
                 marker = 'o', markersize = 2, capsize = 4, c = 'grey')
axes[1].plot(Xi, reg_weighted.params[0]/10**16 + Xi*reg_weighted.params[1]/10**16, c ='navy')
axes[1].set_ylabel(f'{labels[atmos_no]} \n{units[atmos_no]}', size = 10)
axes[1].set_xlabel('Broadleaf Forest Cover %', size = 10)

axes[2].hist(resid_WLS, color = 'grey')
axes[2].set_ylabel('Residual count', size = 10)
axes[2].set_xlabel('Isoprene residuals for the WLS regression', size = 10)


# axes.tick_params(axis='both', which='major', labelsize=14)
fig.text(0.23, 0.89, f'y = {reg.coef_[1]:.2e}*x + {reg.coef_[0]:.2e}  \nR$^{2}$ = {score:.2f}', size = 12, c = 'navy')
fig.text(0.23, 0.57, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = {reg_weighted.rsquared:.2f}', size = 12, c = 'navy')

fig.text(0.04, 0.95, '(a)', size = 14)
fig.text(0.04, 0.65, '(b)', size = 14)
fig.text(0.04, 0.33, '(c)', size = 14)

fig.tight_layout()

# save figure
fig.savefig(f'C:/Users/s2261807/Documents/GitHub/SouthernAmazon_figures/{labels[atmos_no]}_wet_season_regression_summary.png', dpi = 300)
# 
###


# =============================================================================
# Robust regression, all data
# =============================================================================

### LAI ###
#------------------------------------------------------------------------------
Xi = data_pd['LAI']
# Xi = data_pd[['LC', 'Fire']]
y = data_pd['Atmos']

X = sm.add_constant(Xi)

reg = TheilSenRegressor().fit(X, y)

score = reg.score(X, y)
print(score)
y_pred = reg.predict(X)

plt.figure()
plt.scatter(Xi, y, color="indigo", marker="x", s=40, label = 'Data')
plt.plot(Xi, y_pred, label = 'Predicted')


### Broadleaf Forest ###
#------------------------------------------------------------------------------
Xi = data_pd['LC']
# Xi = data_pd[['LC', 'Fire']]
y = data_pd['Atmos']

X = sm.add_constant(Xi)

reg = TheilSenRegressor().fit(X, y)
# reg = OLS().fit(X, y)

score = reg.score(X, y)
print(score)
y_pred = reg.predict(X)

plt.figure()
plt.scatter(Xi, y, color="indigo", marker="x", s=40, label = 'Data')
plt.plot(Xi, y_pred, label = 'Predicted')


### Burned area ###
#------------------------------------------------------------------------------
Xi = data_pd['Fire']
# Xi = data_pd[['LC', 'Fire']]
y = data_pd['Atmos']

X = sm.add_constant(Xi)

reg = TheilSenRegressor().fit(X, y)

score = reg.score(X, y)
print(score)
y_pred = reg.predict(X)

plt.figure()
plt.scatter(Xi, y, color="indigo", marker="x", s=40, label = 'Data')
plt.plot(Xi, y_pred, label = 'Predicted')



# =============================================================================
# data exploration figures
# =============================================================================
# NO2 ~ log(fire)
Xi = data_pd['Fire_log']
y = data_pd['Atmos']
X = sm.add_constant(Xi)

reg = TheilSenRegressor().fit(X, y)

score = reg.score(X, y)
print(score)
y_pred = reg.predict(X)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
axes.scatter(Xi, y, facecolors = 'none', edgecolor="grey", s=40, label = 'Data')
axes.plot(Xi, y_pred, label = 'Predicted', c = 'navy')
axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 16)
axes.set_xlabel('log of burned area [% grid cell area]', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'
fig.text(0.23, 0.7, f'y = {reg.coef_[1]:.2e}*x + {reg.coef_[0]:.2e}  \nR$^{2}$ = {score:.2f}', size = 12, c = 'navy')




# log(no2) ~ log(fire)
Xi = data_pd['Fire_log']
y = data_pd['Atmos_log']
X = sm.add_constant(Xi)

reg = TheilSenRegressor().fit(X, y)

score = reg.score(X, y)
print(score)
y_pred = reg.predict(X)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
axes.scatter(Xi, y, facecolors = 'none', edgecolor="grey", s=40, label = 'Data')
axes.plot(Xi, y_pred, label = 'Predicted', c = 'navy')
axes.set_ylabel(f'log of {labels[atmos_no]} {units[atmos_no]}', size = 16)
axes.set_xlabel('log of burned area [% grid cell area]', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'
fig.text(0.23, 0.7, f'y = {reg.coef_[1]:.2e}*x + {reg.coef_[0]:.2e}  \nR$^{2}$ = {score:.2f}', size = 12, c = 'navy')



# no2 ~ sqrt(fire)
Xi = data_pd['Fire_sqrt']
y = data_pd['Atmos']
X = sm.add_constant(Xi)

reg = TheilSenRegressor().fit(X, y)

score = reg.score(X, y)
print(score)
y_pred = reg.predict(X)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
axes.scatter(Xi, y, facecolors = 'none', edgecolor="grey", s=40, label = 'Data')
axes.plot(Xi, y_pred, label = 'Predicted', c = 'navy')
axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 16)
axes.set_xlabel('Sqrt of burned area [% grid cell area]', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'
fig.text(0.6, 0.7, f'y = {reg.coef_[1]:.2e}*x + {reg.coef_[0]:.2e}  \nR$^{2}$ = {score:.2f}', size = 12, c = 'navy')













# =============================================================================
# use statsmodels: https://www.statsmodels.org/stable/index.html for multiple linear regression
# =============================================================================

# from statsmodels.formula.api import  ols
# Xi = data_pd['LAI']
Xi = data_pd['LC']
# Xi = data_pd[['LC', 'Fire']]
# Xi = data_pd['Fire']
# Xi = data_pd[['Fire', 'LAI']]
y = data_pd['Atmos']

X = sm.add_constant(Xi)

mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())

y_pred = res.predict(X)

plt.figure()
plt.scatter(Xi, y, color="indigo", marker="x", s=40, label = 'Data')
plt.plot(Xi, y_pred, label = 'Predicted')


from statsmodels.formula.api import  rlm #ols,
# Xi = data_pd['LAI']
Xi = data_pd[['LAI', 'Fire']]
# Xi = data_pd['Fire']
# Xi = data_pd[['Fire', 'LAI']]
y = data_pd['Atmos']

X = sm.add_constant(Xi)

mod = rlm("Atmos ~ LAI + Fire", data_pd).fit()
print(mod.summary())



# resid = np.zeros(y.shape[0])
# for i in range(y.shape[0]):
#     resid[i] = y[i] - (res.params[0] + Xi[i]*res.params[1])

# for i in range(20):
#     resid[i] = data3_means[i] - (res.params[0] + np.log(lcs[i])*res.params[1])

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

# axes.scatter(Xi, resid, marker = 'x', c = 'grey')

axes.tick_params(axis='both', which='major', labelsize=14)


# axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]} residual', size = 16)
# axes.set_xlabel('Burned area', size = 16)

# =============================================================================
# scatter plot with regression line
# =============================================================================

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

axes.scatter(Xi, y, marker = 'x', c = 'grey')
axes.plot(Xi, res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')

axes.tick_params(axis='both', which='major', labelsize=14)

fig.text(0.3, 0.75, f'y = {res.params[1]:.2e}*x + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 16)
axes.set_xlabel('Broadleaf Forest Cover %', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'

# save figure
# fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/{labels[atmos_no]}_regression_broadleaf_wet.png', dpi = 300)



# =============================================================================
# mean values regression - broadleaf
# =============================================================================
# dry season all
data3 = []
data3_len = []
data4 = []
data4_len = []

for p in range(5, 101, 5): # or 51 for urban
    distribution_data = get_atm_dist(atmos_dry_slice, reshape_lc, p)
    distribution_data = distribution_data[~np.isnan(distribution_data)]
    data3.append(distribution_data)
    data3_len.append(len(distribution_data))
    
for p in range(5, 101, 5): # or 51 for urban
    distribution_data = get_atm_dist(fire_slice, reshape_lc, p)
    distribution_data = distribution_data[~np.isnan(distribution_data)]
    data4.append(distribution_data)
    data4_len.append(len(distribution_data))
    
def calc_statistic2(data, stat = 'median'):
    data_means = np.zeros(20)
    if stat == 'mean':
        for i,a in enumerate(data):
            data_means[i]=a.mean()
    elif stat == 'median':
        for i,a in enumerate(data):
            data_means[i]=np.median(a)  
    elif stat == 'std':
        for i,a in enumerate(data):
            data_means[i]=np.std(a)          
    else:
        print('invalid stat')
    return data_means
        
data3_means = calc_statistic2(data3, stat = 'mean')
data4_means = calc_statistic2(data4, stat= 'mean')
data3_std = calc_statistic2(data3, stat = 'std')
data4_std = calc_statistic2(data4, stat= 'std')
data3_err = calc_statistic2(data3, stat = 'std')/np.sqrt(data3_len)
data4_err = calc_statistic2(data4, stat= 'std')/np.sqrt(data4_len)
lcs = np.arange(2.5, 98, 5)


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
axes.errorbar(lcs, data3_means, data3_err, linestyle = '', marker = '*', capsize = 4)
axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 16)
axes.set_xlabel('Broadleaf Forest Cover %', size = 16)
# plt.scatter(lcs, data3_means)
# plt.scatter(data4_means, data3_means)

data = {'Atmos': data3_means, 'LC': lcs, 'LC_log': np.log(lcs), 'Fire': data4_means} #'LAI': lai_1d, } #, 'expFire': np.exp(fire_1d)}
data_pd = pd.DataFrame(data)

Xi = data_pd['LC']
# Xi = data_pd[['LC', 'Fire']]
y = data_pd['Atmos']

X = sm.add_constant(Xi)

mod = sm.OLS(y, X, missing = 'drop')

res = mod.fit()

print(res.summary())

### weighted least squares

mod_weighted = sm.WLS(y, X, weights = 1/data3_std, missing = 'drop')
# mod_weighted = sm.WLS(y, X, weights = 1/np.log(data3_std), missing = 'drop')
reg_weighted = mod_weighted.fit()
print(reg_weighted.summary())

### visualisation OLS and WLS
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

axes.errorbar(Xi, y,  data3_err, linestyle = '', marker = 'o', capsize = 4, c = 'grey')
axes.plot(Xi, res.params[0] + Xi*res.params[1], c = 'navy', ls = ':', label = 'OLS')
axes.plot(Xi, reg_weighted.params[0] + Xi*reg_weighted.params[1], c = 'maroon', ls = ':', label = 'WLS')
# axes.errorbar(np.exp(Xi), y, data3_err, linestyle = '', marker = 'o', c = 'grey')
# axes.plot(np.exp(Xi), res.params[0] + Xi*res.params[1], c = 'navy', ls = ':', label = 'OLS')
# axes.plot(np.exp(Xi), reg_weighted.params[0] + Xi*reg_weighted.params[1], c = 'maroon', ls = ':', label = 'WLS')

# axes.errorbar(np.exp(Xi), y, data3_err, linestyle = '', marker = 'o', c = 'grey')
# # axes.plot(np.exp(Xi), res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')
# axes.plot(np.exp(Xi), reg_weighted.params[0] + Xi*reg_weighted.params[1], c = 'maroon', ls = ':', label = 'WLS')

axes.tick_params(axis='both', which='major', labelsize=14)
axes.legend(loc='upper left')
fig.text(0.5, 0.25, f'y = {res.params[1]:.2e}*x + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14, c = 'navy')
fig.text(0.5, 0.15, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = {reg_weighted.rsquared:.2f}', size = 14, c = 'maroon')
# fig.text(0.5, 0.15, f'y = {res.params[1]:.2e}*log(x) + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 16)
axes.set_xlabel('Broadleaf Forest Cover %', size = 16)


# save figure
# fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/{labels[atmos_no]}_OLSvsWLS_broadleaf_dry_means.png', dpi = 300)
# 
###
resid = np.zeros(20)

for i in range(20):
    resid[i] = data3_means[i] - (res.params[0] + Xi[i]*res.params[1])

resid_WLS = np.zeros(20)

for i in range(20):
    resid_WLS[i] = data3_means[i] - (reg_weighted.params[0] + Xi[i]*reg_weighted.params[1])
# for i in range(20):
#     resid[i] = data3_means[i] - (res.params[0] + np.log(lcs[i])*res.params[1])

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

axes.scatter(lcs, resid, marker = 'x', c = 'navy')
axes.scatter(lcs, resid_WLS, marker = 'o', facecolors = 'none', edgecolors = 'maroon')

axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]} residual', size = 16)
axes.set_xlabel('Broadleaf Forest Cover %', size = 16)

# save figure
# fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/{labels[atmos_no]}_regression_logbroadleaf2_dry_means.png', dpi = 300)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

# axes.hist(resid)
axes.hist(resid_WLS)

axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_ylabel('Count', size = 16)
axes.set_xlabel('Residual', size = 16)

# save figure
# fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/{labels[atmos_no]}_regression_logbroadleaf2_dry_means.png', dpi = 300)


# =============================================================================
# multiple regressors
# =============================================================================
data = {'Atmos': data3_means, 'LC': lcs, 'LC_log': np.log(lcs), 'Fire': data4_means} #'LAI': lai_1d, } #, 'expFire': np.exp(fire_1d)}


data_pd = pd.DataFrame(data)

# Xi = data_pd['LC_log']
Xi = data_pd[['LC_log', 'Fire']]
y = data_pd['Atmos']

X = sm.add_constant(Xi)

mod = sm.OLS(y, X, missing = 'drop')

res = mod.fit()

print(res.summary())

### weighted least squares

mod_weighted = sm.WLS(y, X, weights = 1/data3_std, missing = 'drop')
# mod_weighted = sm.WLS(y, X, weights = 1/np.log(data3_std), missing = 'drop')
reg_weighted = mod_weighted.fit()
print(reg_weighted.summary())

### visualisation OLS and WLS
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

# # axes.errorbar(Xi, y,  data3_err, linestyle = '', marker = 'o', capsize = 4, c = 'grey')
# # axes.plot(Xi, res.params[0] + Xi*res.params[1], c = 'navy', ls = ':', label = 'OLS')
# # axes.plot(Xi, reg_weighted.params[0] + Xi*reg_weighted.params[1], c = 'maroon', ls = ':', label = 'WLS')
# axes.errorbar(np.exp(Xi), y, data3_err, linestyle = '', marker = 'o', c = 'grey')
# axes.plot(np.exp(Xi), res.params[0] + Xi*res.params[1], c = 'navy', ls = ':', label = 'OLS')
# axes.plot(np.exp(Xi), reg_weighted.params[0] + Xi*reg_weighted.params[1], c = 'maroon', ls = ':', label = 'WLS')

# # axes.errorbar(np.exp(Xi), y, data3_err, linestyle = '', marker = 'o', c = 'grey')
# # # axes.plot(np.exp(Xi), res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')
# # axes.plot(np.exp(Xi), reg_weighted.params[0] + Xi*reg_weighted.params[1], c = 'maroon', ls = ':', label = 'WLS')

# axes.tick_params(axis='both', which='major', labelsize=14)
# axes.legend(loc='upper left')
# fig.text(0.5, 0.25, f'y = {res.params[1]:.2e}*x + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14, c = 'navy')
# fig.text(0.5, 0.15, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = {reg_weighted.rsquared:.2f}', size = 14, c = 'maroon')
# # fig.text(0.5, 0.15, f'y = {res.params[1]:.2e}*log(x) + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

# axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 16)
# axes.set_xlabel('Broadleaf Forest Cover %', size = 16)


# save figure
# fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/{labels[atmos_no]}_OLSvsWLS_broadleaf_dry_means.png', dpi = 300)
# 
###
resid = np.zeros(20)

for i in range(20):
    resid[i] = data3_means[i] - (res.params[0] + Xi['LC_log'][i]*res.params[1]+ Xi['Fire'][i]*res.params[2])

resid_WLS = np.zeros(20)

for i in range(20):
    resid_WLS[i] = data3_means[i] - (reg_weighted.params[0] + Xi['LC_log'][i]*reg_weighted.params[1]+ Xi['Fire'][i]*reg_weighted.params[2])
# for i in range(20):
#     resid[i] = data3_means[i] - (res.params[0] + np.log(lcs[i])*res.params[1])

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

axes.scatter(lcs, resid, marker = 'x', c = 'navy')
axes.scatter(lcs, resid_WLS, marker = 'o', facecolors = 'none', edgecolors = 'maroon')

axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]} residual', size = 16)
axes.set_xlabel('Broadleaf Forest Cover %', size = 16)

# save figure
# fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/{labels[atmos_no]}_regression_logbroadleaf2_dry_means.png', dpi = 300)

# =============================================================================
# Observed vs modelled values
# =============================================================================


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

axes.scatter(y, res.params[0] + Xi*res.params[1], marker = 'x', c = 'grey', label = f'{labels[atmos_no]} {units[atmos_no]}')
axes.plot(np.linspace(y.min(), y.max(), 50), np.linspace(y.min(), y.max(), 50), c = 'navy', lw = 0.5)
# axes.scatter(np.exp(Xi), y, marker = 'x', c = 'grey')
# axes.plot(np.exp(Xi), res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')

axes.tick_params(axis='both', which='major', labelsize=14)
axes.legend()

fig.text(0.4, 0.15, f'y = {res.params[1]:.2e}*x + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 10)
# fig.text(0.5, 0.15, f'y = {res.params[1]:.2e}*log(x) + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

axes.set_xlabel('Observed', size = 16)
axes.set_ylabel('Modelled', size = 16)


### log version

Xi = data_pd['LC_log']
# Xi = data_pd[['LC', 'Fire']]
y = data_pd['Atmos']

X = sm.add_constant(Xi)

mod = sm.OLS(y, X, missing = 'drop')

res = mod.fit()
print(res.summary())

z = np.exp(Xi)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

axes.scatter(y, res.params[0] + Xi*res.params[1], marker = 'x', c = 'grey', label = f'{labels[atmos_no]} {units[atmos_no]}')
axes.plot(np.linspace(y.min(), y.max(), 50), np.linspace(y.min(), y.max(), 50), c = 'navy', lw = 0.5)
# axes.scatter(np.exp(Xi), y, marker = 'x', c = 'grey')
# axes.plot(np.exp(Xi), res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')

axes.tick_params(axis='both', which='major', labelsize=14)
axes.legend()

fig.text(0.4, 0.15, f'y = {res.params[1]:.2e}*log(landcover) + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 10)
# fig.text(0.5, 0.15, f'y = {res.params[1]:.2e}*log(x) + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

axes.set_xlabel('Observed', size = 16)
axes.set_ylabel('Modelled', size = 16)

# =============================================================================
# mean values regression - lai
# =============================================================================
# dry season all
data3 = []
data3_len = []
data4 = []
data4_len = []

for p in range(5, 71, 5): # or 51 for urban
    distribution_data = get_atm_dist(atmos_dry_slice, reshape_lc, p)
    distribution_data = distribution_data[~np.isnan(distribution_data)]
    data3.append(distribution_data)
    data3_len.append(len(distribution_data))
    
for p in range(5, 71, 5): # or 51 for urban
    distribution_data = get_atm_dist(fire_slice, reshape_lc, p)
    distribution_data = distribution_data[~np.isnan(distribution_data)]
    data4.append(distribution_data)
    data4_len.append(len(distribution_data))
    
def calc_statistic2(data, stat = 'median'):
    data_means = np.zeros(14)
    if stat == 'mean':
        for i,a in enumerate(data):
            data_means[i]=a.mean()
    elif stat == 'median':
        for i,a in enumerate(data):
            data_means[i]=np.median(a)  
    else:
        print('invalid stat')
    return data_means
        
data3_means = calc_statistic2(data3, stat = 'mean')
data4_means = calc_statistic2(data4, stat= 'mean')
lais = np.arange(2.5, 71, 5)

# plt.scatter(lcs, data3_means)
# plt.scatter(data4_means, data3_means)

data = {'Atmos': data3_means, 'LAI': lais, 'LAI_log': np.log(lais), 'Fire': data4_means} #'LAI': lai_1d, } #, 'expFire': np.exp(fire_1d)}
data_pd = pd.DataFrame(data)

Xi = data_pd['LAI']
# Xi = data_pd[['LC', 'Fire']]
y = data_pd['Atmos']

X = sm.add_constant(Xi)

mod = sm.OLS(y, X, missing = 'drop')

res = mod.fit()

print(res.summary())


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

axes.scatter(Xi, y, marker = 'x', c = 'grey')
axes.plot(Xi, res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')
# axes.scatter(np.exp(Xi), y, marker = 'x', c = 'grey')
# axes.plot(np.exp(Xi), res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')

axes.tick_params(axis='both', which='major', labelsize=14)

fig.text(0.5, 0.15, f'y = {res.params[1]:.2e}*x + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)
# fig.text(0.5, 0.15, f'y = {res.params[1]:.2e}*log(x) + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 16)
axes.set_xlabel('LAI', size = 16)

# save figure
# fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/{labels[atmos_no]}_regression_broadleaf_dry_means.png', dpi = 300)

resid = np.zeros(14)

for i in range(14):
    resid[i] = data3_means[i] - (res.params[0] + lcs[i]*res.params[1])

# for i in range(20):
#     resid[i] = data3_means[i] - (res.params[0] + np.log(lcs[i])*res.params[1])

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

axes.scatter(data4_means, resid, marker = 'x', c = 'grey')

axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]} residual', size = 16)
axes.set_xlabel('Burned area', size = 16)

# save figure
# fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/{labels[atmos_no]}_regression_logbroadleaf2_dry_means.png', dpi = 300)

# =============================================================================
# Observed vs modelled values
# =============================================================================


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

axes.scatter(y, res.params[0] + Xi*res.params[1], marker = 'x', c = 'grey', label = f'{labels[atmos_no]} {units[atmos_no]}')
axes.plot(np.linspace(y.min(), y.max(), 50), np.linspace(y.min(), y.max(), 50), c = 'navy', lw = 0.5)
# axes.scatter(np.exp(Xi), y, marker = 'x', c = 'grey')
# axes.plot(np.exp(Xi), res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')

axes.tick_params(axis='both', which='major', labelsize=14)
axes.legend()

fig.text(0.4, 0.15, f'y = {res.params[1]:.2e}*x + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 10)
# fig.text(0.5, 0.15, f'y = {res.params[1]:.2e}*log(x) + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

axes.set_xlabel('Observed', size = 16)
axes.set_ylabel('Modelled', size = 16)


### log version

Xi = data_pd['LAI_log']
# Xi = data_pd[['LC', 'Fire']]
y = data_pd['Atmos']

X = sm.add_constant(Xi)

mod = sm.OLS(y, X, missing = 'drop')

res = mod.fit()
print(res.summary())
z = np.exp(Xi)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

axes.scatter(y, res.params[0] + Xi*res.params[1], marker = 'x', c = 'grey', label = f'{labels[atmos_no]} {units[atmos_no]}')
axes.plot(np.linspace(y.min(), y.max(), 50), np.linspace(y.min(), y.max(), 50), c = 'navy', lw = 0.5)
# axes.scatter(np.exp(Xi), y, marker = 'x', c = 'grey')
# axes.plot(np.exp(Xi), res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')

axes.tick_params(axis='both', which='major', labelsize=14)
axes.legend()

fig.text(0.4, 0.15, f'y = {res.params[1]:.2e}*log(LAI) + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 10)
# fig.text(0.5, 0.15, f'y = {res.params[1]:.2e}*log(x) + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

axes.set_xlabel('Observed', size = 16)
axes.set_ylabel('Modelled', size = 16)



# =============================================================================
# mean values regression - fire
# =============================================================================
# dry season all
data3 = []
data3_len = []

data4 = []
data4_len = []

# def get_fire_dist25(atm_comp, lc, perc):
#     if perc == 0.000025:
#         cover = ma.masked_less_equal(lc, perc).mask.flatten()
#     if perc ==0.05:
#         cover = ma.masked_greater(lc, 0.0475).mask.flatten()
#     else:
#         cover = ma.masked_inside(lc, perc - 0.0249, perc).mask.flatten()
    
#     atm_comp_1d = atm_comp.values.flatten()   
#     # atm_comp_1d = atm_comp.flatten()   # use for hcho
#     if np.any(cover) == False:
#         cover = np.zeros_like(atm_comp_1d, dtype = np.bool)
#     atm_comp_dist = atm_comp_1d[cover]
#     return atm_comp_dist

# for p in range(0.000025, 0.0501, 0.000025): # or 51 for urban
#     distribution_data = get_fire_dist25(atmos_dry_slice, fire_slice, p)
#     distribution_data = distribution_data[~np.isnan(distribution_data)]
#     data3.append(distribution_data)
#     data3_len.append(len(distribution_data))


def get_fire_dist25(atm_comp, lc, perc):
    if perc == 25:
        cover = ma.masked_less_equal(lc, perc).mask.flatten()
    if perc ==500:
        cover = ma.masked_greater(lc, 475).mask.flatten()
    else:
        cover = ma.masked_inside(lc, perc - 24.9, perc).mask.flatten()
    
    atm_comp_1d = atm_comp.values.flatten()
    fire_1d = lc.values.flatten()
    # atm_comp_1d = atm_comp.flatten()   # use for hcho
    if np.any(cover) == False:
        cover = np.zeros_like(atm_comp_1d, dtype = np.bool)
    atm_comp_dist = atm_comp_1d[cover]
    fire_dist = fire_1d[cover]
    return atm_comp_dist, fire_dist

for p in range(25, 501, 25): # or 51 for urban
    distribution_data = get_fire_dist25(atmos_dry_slice, fire_slice*10000, p)
    atm_data1 = distribution_data[0]
    atm_data = atm_data1[~np.isnan(atm_data1)]
    fire_data = distribution_data[1]
    fire_data = fire_data[~np.isnan(atm_data1)]
    data3.append(atm_data)
    data4.append(fire_data)
    data3_len.append(len(atm_data))
    data4_len.append(len(fire_data))
    
# def get_fire_dist_2(atm_comp, lc, perc):
#     if perc == 25:
#         cover = ma.masked_less_equal(lc, perc).mask.flatten()
#     if perc ==500:
#         cover = ma.masked_greater(lc, 475).mask.flatten()
#     else:
#         cover = ma.masked_inside(lc, perc - 24.9, perc).mask.flatten()
#     atm_comp_1d = atm_comp.flatten()   # use for hcho
#     if np.any(cover) == False:
#         cover = np.zeros_like(atm_comp_1d, dtype = np.bool)
#     atm_comp_dist = atm_comp_1d[cover]
#     return atm_comp_dist

# for p in range(25, 501, 25): # or 51 for urban
#     distribution_data = get_fire_dist_2(reshape_lc, fire_slice*10000, p)
#     distribution_data = distribution_data[~np.isnan(distribution_data)]
#     data4.append(distribution_data)
#     data4_len.append(len(distribution_data))

    
def calc_statistic2(data, stat = 'median'):
    data_means = np.zeros(20)
    if stat == 'mean':
        for i,a in enumerate(data):
            data_means[i]=a.mean()
    elif stat == 'median':
        for i,a in enumerate(data):
            data_means[i]=np.median(a)  
    elif stat == 'std':
        for i,a in enumerate(data):
            data_means[i]=np.std(a)          
    else:
        print('invalid stat')
    return data_means
        
data3_means = calc_statistic2(data3, stat = 'mean')
data4_means = calc_statistic2(data4, stat= 'mean')
data3_std = calc_statistic2(data3, stat = 'std')
data4_std = calc_statistic2(data4, stat= 'std')
data3_err = calc_statistic2(data3, stat = 'std')/np.sqrt(data3_len)
data4_err = calc_statistic2(data4, stat= 'std')/np.sqrt(data4_len)

# fires = np.arange(12.5/10000, 500/10000, 25/10000)

# plt.scatter(lcs, data3_means)
# plt.scatter(data4_means, data3_means)

data = {'Atmos': data3_means, 'Atmos_log':np.log(data3_means), 'Fire': data4_means/10000, 'Fire_log': np.log(data4_means/10000), 'Fire_sqrt' : np.sqrt(data4_means/10000)} #'LAI': lai_1d, } #, 'expFire': np.exp(fire_1d)}
data_pd = pd.DataFrame(data)

Xi = data_pd['Fire_log']
# Xi = data_pd[['LC', 'Fire']]
y = data_pd['Atmos_log']

X = sm.add_constant(Xi)

### weighted least squares

mod_weighted = sm.WLS(y, X, weights = 1/data3_err, missing = 'drop')
reg_weighted = mod_weighted.fit()
print(reg_weighted.summary())
y_pred = reg_weighted.predict(X)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
axes.scatter(Xi, y, marker = 'o', facecolors = 'none', edgecolor = 'grey')
axes.plot(Xi, y_pred, label = 'Predicted', c = 'navy')
axes.set_ylabel(f'Nat log of mean {labels[atmos_no]} {units[atmos_no]}', size = 16)
axes.set_xlabel('Nat log of mean burned area [% grid cell area]', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'
fig.text(0.3, 0.7, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = 0.98', size = 12, c = 'navy')
# axes.set_ylim(ymin = 34.5, ymax = 36)




fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
axes.errorbar(Xi, y,  np.log(data3_err), linestyle = '', marker = 'o', capsize = 4, c = 'grey')
axes.plot(Xi, y_pred, label = 'Predicted', c = 'navy')
axes.set_ylabel(f'Log of {labels[atmos_no]} {units[atmos_no]}', size = 16)
axes.set_xlabel('Log of burned area [% grid cell area]', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'
fig.text(0.3, 0.7, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = 0.94', size = 12, c = 'navy')
# axes.set_ylim(ymin = 34.5, ymax = 36)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
axes.scatter(Xi, y, marker = 'o', facecolors = 'none', edgecolor = 'grey')
axes.plot(Xi, y_pred, label = 'Predicted', c = 'navy')
axes.set_ylabel(f'Nat log of mean {labels[atmos_no]} {units[atmos_no]}', size = 16)
axes.set_xlabel('Nat log of mean burned area [% grid cell area]', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'
fig.text(0.3, 0.7, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = 0.98', size = 12, c = 'navy')
# axes.set_ylim(ymin = 34.5, ymax = 36)



fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
axes.errorbar(data['Fire'], data['Atmos'],  data3_err, linestyle = '', marker = 'o', capsize = 4, c = 'grey')
axes.plot(Xi**2, y_pred, label = 'Predicted', c = 'navy')
axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 16)
axes.set_xlabel('Burned area [% grid cell area]', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'
fig.text(0.3, 0.7, f'y = {reg_weighted.params[1]:.2e}*sqrt(x) + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = 0.97', size = 12, c = 'navy')
# axes.set_ylim(ymin = 34.5, ymax = 36)


### visualisation OLS and WLS
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

# axes.errorbar(Xi, y,  data3_err, linestyle = '', marker = 'o', capsize = 4, c = 'grey')
# axes.plot(Xi, res.params[0] + Xi*res.params[1], c = 'navy', ls = ':', label = 'OLS')
# axes.plot(Xi, reg_weighted.params[0] + Xi*reg_weighted.params[1], c = 'maroon', ls = ':', label = 'WLS')
axes.errorbar(Xi, y, np.log(data3_err), linestyle = '', marker = 'o', capsize = 4, c = 'grey')
axes.plot(Xi, res.params[0] + Xi*res.params[1], c = 'navy', ls = ':', label = 'OLS')
axes.plot(Xi, reg_weighted.params[0] + Xi*reg_weighted.params[1], c = 'maroon', ls = ':', label = 'WLS')
axes.set_ylim(ymin = -0.01*10**14, ymax = 0.01*10**14)
# axes.errorbar(np.exp(Xi), y, data3_err, linestyle = '', marker = 'o', c = 'grey')
# # axes.plot(np.exp(Xi), res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')
# axes.plot(np.exp(Xi), reg_weighted.params[0] + Xi*reg_weighted.params[1], c = 'maroon', ls = ':', label = 'WLS')

axes.tick_params(axis='both', which='major', labelsize=14)
axes.legend(loc='upper left')
# fig.text(0.5, 0.25, f'y = {res.params[1]:.2e}*x + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14, c = 'navy')
# fig.text(0.5, 0.15, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = {reg_weighted.rsquared:.2f}', size = 14, c = 'maroon')
fig.text(0.5, 0.25, f'y = {res.params[1]:.2e}*x + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14, c = 'navy')
fig.text(0.5, 0.15, f'y = {reg_weighted.params[1]:.2e}*x + {reg_weighted.params[0]:.2e}  \nR$^{2}$ = {reg_weighted.rsquared:.2f}', size = 14, c = 'maroon')

axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 16)
axes.set_xlabel('Burned area % x 10^4', size = 16)


# save figure
# fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/{labels[atmos_no]}_OLSvsWLS_broadleaf_dry_means.png', dpi = 300)
# 
###
# resid = np.zeros(20)

# for i in range(20):
#     resid[i] = data3_means[i] - (res.params[0] + Xi[i]*res.params[1])

resid_WLS = np.zeros(20)

for i in range(20):
    resid_WLS[i] = data3_means[i] - (reg_weighted.params[0] + Xi[i]*reg_weighted.params[1])

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
axes.hist(resid_WLS, color = 'grey')
axes.set_ylabel('Residual count', size = 10)
axes.set_xlabel('NO2 residuals for the WLS regression', size = 10)


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

# axes.scatter(Xi, resid, marker = 'x', c = 'navy')
axes.scatter(np.exp(Xi), resid_WLS, marker = 'o', facecolors = 'none', edgecolors = 'maroon')

axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]} residual', size = 16)
axes.set_xlabel('Burned area % x 10^4', size = 16)

# save figure
# fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/{labels[atmos_no]}_regression_logbroadleaf2_dry_means.png', dpi = 300)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

# axes.hist(resid)
axes.hist(resid_WLS)

axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_ylabel('Count', size = 16)
axes.set_xlabel('Residual', size = 16)

# save figure
# fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/{labels[atmos_no]}_regression_logbroadleaf2_dry_means.png', dpi = 300)


# =============================================================================
# Observed vs modelled values
# =============================================================================


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

axes.scatter(y, res.params[0] + Xi*res.params[1], marker = 'x', c = 'grey', label = f'{labels[atmos_no]} {units[atmos_no]}')
axes.plot(np.linspace(y.min(), y.max(), 50), np.linspace(y.min(), y.max(), 50), c = 'navy', lw = 0.5)
# axes.scatter(np.exp(Xi), y, marker = 'x', c = 'grey')
# axes.plot(np.exp(Xi), res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')

axes.tick_params(axis='both', which='major', labelsize=14)
axes.legend()

fig.text(0.4, 0.15, f'y = {res.params[1]:.2e}*x + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 10)
# fig.text(0.5, 0.15, f'y = {res.params[1]:.2e}*log(x) + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

axes.set_xlabel('Observed', size = 16)
axes.set_ylabel('Modelled', size = 16)


### log version

Xi = data_pd['Fire_log']
# Xi = data_pd[['LC', 'Fire']]
y = data_pd['Atmos']

X = sm.add_constant(Xi)

mod = sm.OLS(y, X, missing = 'drop')

res = mod.fit()
print(res.summary())
z = np.exp(Xi)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

axes.scatter(y, res.params[0] + Xi*res.params[1], marker = 'x', c = 'grey', label = f'{labels[atmos_no]} {units[atmos_no]}')
axes.plot(np.linspace(y.min(), y.max(), 50), np.linspace(y.min(), y.max(), 50), c = 'navy', lw = 0.5)
# axes.scatter(np.exp(Xi), y, marker = 'x', c = 'grey')
# axes.plot(np.exp(Xi), res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')

axes.tick_params(axis='both', which='major', labelsize=14)
axes.legend()

fig.text(0.4, 0.15, f'y = {res.params[1]:.2e}*log(burned area) + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 10)
# fig.text(0.5, 0.15, f'y = {res.params[1]:.2e}*log(x) + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

axes.set_xlabel('Observed', size = 16)
axes.set_ylabel('Modelled', size = 16)

# # =============================================================================
# # ### 2 variables - add after mean broadleaf
# # =============================================================================

# # Xi = data_pd['Fire']
# Xi = data_pd[['LC', 'Fire']]
# y = data_pd['Atmos']

# X = sm.add_constant(Xi)

# mod = sm.OLS(y, X, missing = 'drop')

# res = mod.fit()

# print(res.summary())



# # =============================================================================
# # scatter plot with regression line
# # =============================================================================

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

# axes.scatter(Xi, y, marker = 'x', c = 'grey')
# axes.plot(Xi, res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')

# axes.tick_params(axis='both', which='major', labelsize=14)

# fig.text(0.3, 0.75, f'y = {res.params[1]:.2e}*x + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

# axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 16)
# axes.set_xlabel('Broadleaf Forest Cover %', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'



# # =============================================================================
# # fire data split for AOD, CO and HCHO
# # =============================================================================
# for i, y in enumerate(lcs):
#     lc = y
    
#     ### keeping only data for months where both datasets are available
#     min_time = xr.DataArray(data = [fire_dry.time.min().values, atmos_dry.time.min().values]).max().values
#     min_year = min_time.astype(str).split('-')[0]
#     max_time = xr.DataArray(data = [fire_dry.time.max().values, atmos_dry.time.max().values]).min().values
#     max_year = max_time.astype(str).split('-')[0]
    
#     fire_slice = fire_dry.sel(time=slice(min_year, max_year))
#     fire_wet_slice = fire_wet.sel(time=slice(min_year, max_year))
#     atmos_dry_slice = atmos_dry.sel(time=slice(min_year, max_year))
#     atmos_wet_slice = atmos_wet.sel(time=slice(min_year, max_year))
#     lc_slice = lc.sel(time=slice(min_year, max_year))
#     lai_slice = lai.sel(time=slice(min_year, max_year))
#     #lc_int = (np.rint(lc_slice*100)).astype(int)
#     lc_int = np.rint(lc_slice) #if LC B used
    
#     ### separating by fire size
#     boundary = 0.015 #0.02 for AOD and CO? 0.015 for HCHO?
#     no_fire = ma.masked_less_equal(fire_slice, boundary).mask
#     yes_fire = ma.masked_greater(fire_slice, boundary).mask
        
#     atmos_dry_fire, atmos_dry_low_fire = mask_data(atmos_dry_slice, yes_fire, no_fire)
#     high_fire, low_fire = mask_data(fire_slice, yes_fire, no_fire)
    
#     reshape_lc = np.zeros_like(atmos_dry_slice)
#     for a in range(lc_int.shape[0]):
#         for b in range(3):
#             reshape_lc[a*3+b,:,:] = lc_int[a]
    
#     reshape_lai = np.zeros_like(atmos_dry_slice)
#     for a in range(lai_slice.shape[0]):
#         for b in range(3):
#             reshape_lai[a*3+b,:,:] = lai_slice[a]

# # =============================================================================
# # check linearity of relationships
# # =============================================================================
# # atmos_1d = atmos_wet_slice.values.ravel()
# # atmos_1d = atmos_dry_slice.values.ravel()
# # atmos_1d = np.where(atmos_dry_1d < 10**17, atmos_dry_1d, np.nan) #for HCHO

# # atmos_1d = atmos_dry_fire.values.ravel()
# atmos_1d = atmos_dry_low_fire.values.ravel()
# atmos_1d = np.where(atmos_1d < 10**17, atmos_1d, np.nan) #for HCHO

# lai_1d = reshape_lai.ravel()

# lc_1d = reshape_lc.ravel()


# # fire_1d = high_fire.values.ravel()
# fire_1d = low_fire.values.ravel()

# plt.scatter(lai_1d, atmos_1d)
# plt.scatter(lc_1d, atmos_1d)
# plt.scatter(fire_1d, atmos_1d)

# plt.scatter(np.exp(fire_1d), atmos_1d)

# ### create pandas dataframe to hold data of interest

# data = {'Atmos': atmos_1d, 'LC': lc_1d, 'LAI': lai_1d, 'Fire': fire_1d} #, 'expFire': np.exp(fire_1d)}

# data_pd = pd.DataFrame(data)

# # =============================================================================
# # use statsmodels: https://www.statsmodels.org/stable/index.html for multiple linear regression
# # =============================================================================


# Xi = data_pd['Fire']
# # Xi = data_pd[['LC', 'Fire']]
# y = data_pd['Atmos']

# X = sm.add_constant(Xi)

# mod = sm.OLS(y, X, missing = 'drop')

# res = mod.fit()

# print(res.summary())


# # =============================================================================
# # scatter plot with regression line
# # =============================================================================

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

# axes.scatter(Xi, y, marker = 'x', c = 'grey')
# axes.plot(Xi, res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')

# axes.tick_params(axis='both', which='major', labelsize=14)

# fig.text(0.3, 0.75, f'y = {res.params[1]:.2e}*x + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

# axes.set_ylabel(f'{labels[atmos_no]} {units[atmos_no]}', size = 16)
# axes.set_xlabel('Burned Area [% grid cell area]', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'

# # save figure
# # fig.savefig(f'M:/figures/atm_chem/comparison/SouthAmazon_stats/{labels[atmos_no]}_regression_broadleaf_wet.png', dpi = 300)


# # # =============================================================================
# # # LC vs LAI
# # # =============================================================================

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

# # axes.scatter(lai.values.ravel()/10, lc.values.ravel(), marker = 'x', c = 'grey')
# axes.scatter(lc.values.ravel(), lai.values.ravel()/10, marker = 'x', c = 'grey')

# # axes.plot(Xi, res.params[0] + Xi*res.params[1], c = 'navy', ls = ':')

# axes.tick_params(axis='both', which='major', labelsize=14)

# # fig.text(0.3, 0.75, f'y = {res.params[1]:.2e}*x + {res.params[0]:.2e}  \nR$^{2}$ = {res.rsquared:.2f}', size = 14)

# axes.set_ylabel('LAI', size = 16)
# axes.set_xlabel('Broadleaf Forest Cover %', size = 16) #'Broadleaf Forest Cover %' 'Burned Area [% grid cell area]'

# # # save figure
# # fig.savefig('M:/figures/land_cover/broadleaf_lai_SAM.png', dpi = 300)

# data_test = {'LC': lc.values.ravel(), 'LAI': lai.values.ravel()/10} #, 'expFire': np.exp(fire_1d)}

# data_test_pd = pd.DataFrame(data_test)

# Xi = data_test_pd['LC']
# y = data_test_pd['LAI']

# X = sm.add_constant(Xi)

# mod = sm.OLS(y, X, missing = 'drop')

# res = mod.fit()

# print(res.summary())

# # =============================================================================
# # plot 2 variable function
# # =============================================================================
# from numpy import exp,arange
# from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

# # the function that I'm going to plot
# def z_func(x,y):
#  return res.params[0] + x*res.params[1] + y*res.params[2]
 
# x = np.linspace(fire_1d.min(), fire_1d.max(), 100)
# y = np.linspace(lc_1d.min(), lc_1d.max(), 100)
# X,Y = meshgrid(x, y) # grid of point
# Z = z_func(X, Y) # evaluation of the function on the grid

# im = imshow(Z,cmap=cm.RdBu) # drawing the function
# # adding the Contour lines with labels
# # cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
# clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
# colorbar(im) # adding the colobar on the right
# # latex fashion title
# title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
# show()


# # =============================================================================
# # additional code/notes
# # =============================================================================
# data = np.arange(1, 100)
# data_log = np.log(data)
# data_e = np.exp(data_log)
