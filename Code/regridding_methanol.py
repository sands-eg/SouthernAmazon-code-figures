# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:43:49 2022

@author: s2261807

script for regridding methanol from 0.5 by 0.5 to 1 by 1 degree

code based IDL script 'REGULAR_GRID_INTERPOLATION.PRO' by Richard Pope at the University of Leeds

"""
# import functions
import xarray as xr
import numpy as np
from scipy import interpolate
import numpy.ma as ma
import pandas as pd

fn = 'R:/methanol/ims_methanol.nc'
ds = xr.open_dataset(fn, decode_times=False)
methanol = ds['methanol'][:]
ds.close()

fn = 'R:\modis_lc\mcd12c1_1deg_trend_corrected.nc'
ds = xr.open_dataset(fn, decode_times=False)
trend = ds['coef1'][:]
ds.close()

# =============================================================================
# interpolation from from 0.5 by 0.625 to 1 by 1 degree grid
# =============================================================================

# INPUTS
lons_old = methanol.lon.values
lats_old = methanol.lat.values
lons_new = trend.lon.values
lats_new = trend.lat.values

methanol_masked = methanol.where(methanol != -999.99)

data_old = methanol_masked[:,:,:]

# OUTPUTS
# data_new = original data gridded to new resolution     [n_lons,n_lats]


# initialise vars
n_lons_old=len(lons_old)
n_lats_old=len(lats_old)
n_lons_new=len(lons_new)
n_lats_new=len(lats_new)
mdi=np.nan

# make new grid
data_new=np.zeros((132, n_lats_new, n_lons_new))+mdi

# now interpolate over lon and lat dimensions
tmp_data=np.zeros((132, n_lats_new, n_lons_old))+mdi

for y in range(132):
    for i_lon in range((n_lons_old)):
        # get tmp data
        tmp_tmp_data = data_old[y, :, i_lon]
        
        # now interpolate lats
        interpfunc = interpolate.interp1d(lats_old, tmp_tmp_data, kind='linear', fill_value="extrapolate")
        tmp_int = interpfunc(lats_new)
        
        # # make sure interpolate did not create negative numbers
        # good_data = ma.masked_greater(tmp_int, 0).mask
        # tmp_int[:] = np.where(good_data, tmp_int[:], np.nan)
        
        tmp_data[y, :, i_lon]=tmp_int
        
    # loop over lats and interpolate over lons
    
    for i_lat in range((n_lats_new)):
        # tmp vars
        tmp_tmp_data = tmp_data[y, i_lat,:]
        
        # now interpolate lats
        interpfunc = interpolate.interp1d(lons_old, tmp_tmp_data, kind='linear', fill_value="extrapolate")
        tmp_int = interpfunc(lons_new)
        
        # # make sure interpolate did not create negative numbers
        # good_data = ma.masked_greater(tmp_int, 0).mask
        # tmp_int[:] = np.where(good_data, tmp_int[:], np.nan)
        
        data_new[y, i_lat, :]=tmp_int


# =============================================================================
# calculating and scaling by global burden
# =============================================================================

### old grid 
lon_edges = np.arange(-180.25, 180.25, 0.5)
lat_edges = np.arange(90.25, -90.25, -0.5)


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
print('Surface Area of Earth in km2 = ', np.sum(surface_area_earth, axis = (0,1)))

burden_old = (data_old * (surface_area_earth*10**10)).sum(axis=(1,2))


### new grid
cell_areas_1 = np.zeros((180, 360))
cell_areas_1 = np.load('R:\modis_1_deg_cell_area.npy')

burden_new = np.nansum(data_new * (cell_areas_1*10**10), axis=(1,2))

###
R = burden_old / burden_new

methanol_new = np.ones_like(data_new)
for i in range(132):
    methanol_new[i,:,:] = data_new[i,:,:] * R.values[i]

time_new = pd.date_range('2008-01-01', '2018-12-31', freq = 'MS')

methanol_monthly = xr.DataArray(
    data=methanol_new,
    dims=["time", "lat", "lon"],
    coords=dict(
        lon=lons_new,
        lat=lats_new,
        time=time_new,
    ),
    attrs=dict(
        description="Monthly mean methanol",
        units="ppbv",
    ),
)       

data = methanol_monthly.rename('methanol')
data.to_netcdf('R:/methanol/methanol_1degree_2008-2018.nc')
