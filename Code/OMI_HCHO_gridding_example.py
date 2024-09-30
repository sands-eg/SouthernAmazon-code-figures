#!/usr/bin/env python
# -*- coding: utf-8 -*- 

###import packages
from os import listdir
from os.path import isfile, join
import numpy as np
from datetime import datetime
import pandas as pd
import h5py

from netCDF4 import Dataset
from netCDF4 import date2num

import time as Time

'''
OMI_HCHO_gridding.py

Purpose:

Gridding OMI HCHO swath data onto 1.0 degree x 1.0 degree grid. 
Adapted from omi_hcho_gridding.pro written by Richard Pope (Uni. of Leeds) in May 2022.
Prepared for L2 HCHO data available from NASA (GES DISC).
'''

__author__ = "E Sands"
__email__ = "e.g.sands@ed.ac.uk"

## set year for run
year = 2017 # replace in start_date and nc file name, f'{}' causes issues with SLURM
days_in_year = 365 #remember leap years, replace in date_array

t1 = Time.perf_counter()

## get list of files 
mypath = "/home/users/egsands/sensecdt/users/egsands/OMI_HCHO_NASA"
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#print(files)

## initialise variables
n_files = len(files)
mdi = np.nan

#print(n_files, mdi)

## get list of days
# testing just on 3 -> 2005-01-01 to 2005-01-03
start_date = datetime.strptime('2017-01-01', '%Y-%m-%d')
date_array = (start_date + pd.to_timedelta(np.arange(days_in_year), 'D')).strftime('%Y-%m-%d')
n_days = len(date_array)
#print(date_array, n_days)

## logical removed? - script not planned to be used for anything other than regridding

## get date information from each satellite file name
tmp_list1, tmp_list2, tmp_list = [], [], []
for i in range(0, len(files)):
    tmp_list1.append(files[i][19:23]) # extracting year
    tmp_list2.append(files[i][24:28]) # extracting month and day (mmdd)
    tmp_list.append(tmp_list1[i]+'-'+tmp_list2[i][:2]+'-'+tmp_list2[i][2:]) # creating list in %Y-%m-%d format
#print(tmp_list)


## grid setup
n_lons, n_lats, n_res = 360, 180, 1.0
lon_centre = np.arange(0,n_lons)*n_res - 180.0 + 0.5
lat_centre = np.arange(0,n_lats)*n_res - 90.0 + 0.5
#print(lat_centre)

## hdf file pathways - needs to be later in script?

#pathways in hdf files
group_data = '/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/'
group_geoloc = '/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/'


## create storage arrays
hcho_grid_daily = np.empty((n_days, n_lons, n_lats))
error_grid_daily = np.empty((n_days, n_lons, n_lats))
hcho_grid_daily[:], error_grid_daily[:] = np.nan, np.nan

hcho_grid_daily_rs = np.empty((n_days, n_lons, n_lats))
error_grid_daily_rs = np.empty((n_days, n_lons, n_lats))
hcho_grid_daily_rs[:], error_grid_daily_rs[:] = np.nan, np.nan

### processing the data
for i in range(n_days): # changed to 1 for testing, should be n_days
    print(date_array[i])
    
    ## work out number of file for given day
    date_loc = []
    for a,b in enumerate(tmp_list):
        if b == date_array[i]: date_loc.append(a)
    # print(date_loc)
    cnt = len(date_loc)
    # print(cnt)
    
    if cnt >= 0: # if data exists for given date
        ## get list of files
        tmp_files=[]
        for a in date_loc:
            tmp_files.append(files[a])
        # print(tmp_files)
        
        ## create grid arrays for calculating daily average
        hcho_grid, hcho_cnts = np.zeros((n_lons, n_lats)), np.zeros((n_lons, n_lats))
        err_grid, err_cnts = np.zeros((n_lons, n_lats)), np.zeros((n_lons, n_lats))
        hcho_grid_rs, hcho_cnts_rs = np.zeros((n_lons, n_lats)), np.zeros((n_lons, n_lats))
        err_grid_rs, err_cnts_rs = np.zeros((n_lons, n_lats)), np.zeros((n_lons, n_lats))
        
        ## read in data
        for j in range(cnt): #tmp changed to 2 for testing, shoudl be cnt
            
            ## special case for 1st file
            if j == 0:
                ## open file
                result = h5py.File(mypath+'/'+tmp_files[j], 'r')
                ## read in data
                lons = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/Longitude'][:]
                lats = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/Latitude'][:]
                
                hcho = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/ColumnAmount'][:]
                hcho_rs = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/ReferenceSectorCorrectedVerticalColumn'][:]
                
                cloud = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/AMFCloudFraction'][:]
                flag = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/MainDataQualityFlag'][:]
                xtrack = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/XtrackQualityFlags'][:]
                
                col_unc = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/ColumnUncertainty'][:]
                # etc
                
                ## close file
                result.close()
                
            else:
                ## open file
                result = h5py.File(mypath+'/'+tmp_files[j], 'r')
                ## read in data
                tmp_lons = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/Longitude'][:]
                tmp_lats = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/Latitude'][:]
                
                tmp_hcho = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/ColumnAmount'][:]
                tmp_hcho_rs = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/ReferenceSectorCorrectedVerticalColumn'][:]
                
                tmp_cloud = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/AMFCloudFraction'][:]
                tmp_flag = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/MainDataQualityFlag'][:]
                tmp_xtrack = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/XtrackQualityFlags'][:]
                
                tmp_col_unc = result['/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/ColumnUncertainty'][:]
                # etc
                
                ## close file
                result.close()
                
                ## concatenate variables to store in one large array
                lons = np.concatenate((lons, tmp_lons))
                lats = np.concatenate((lats, tmp_lats))
                hcho = np.concatenate((hcho, tmp_hcho))
                hcho_rs = np.concatenate((hcho_rs, tmp_hcho_rs))
                cloud = np.concatenate((cloud, tmp_cloud))
                flag = np.concatenate((flag, tmp_flag))
                xtrack = np.concatenate((xtrack, tmp_xtrack))
                col_unc = np.concatenate((col_unc, tmp_col_unc))
            
        ## get swath dimension
        n_swath1=len(hcho[:,0])
        n_swath2=len(hcho[0,:])
        
        ## loop over data dimensions
        for c in range(n_swath1):
            for d in range(n_swath2):
                ## find good data = quality control, clouds etc.
                # add if conditions 
                if cloud[c,d] <= 0.2 and flag[c,d] == 0 and hcho[c,d] >= 0 and xtrack[c,d] == 0: 
                    ## find location on grid
                    lon_loc = np.where(abs(lon_centre - lons[c, d]) == abs(lon_centre - lons[c, d]).min())
                    lat_loc = np.where(abs(lat_centre - lats[c, d]) == abs(lat_centre - lats[c, d]).min())
                    
                    # in case of rare retrieval sites equally between points 1st location is taken
                    lon_loc = lon_loc[0]
                    lat_loc = lat_loc[0]
                    
                    ## store data on grid
                    hcho_grid[lon_loc,lat_loc]=hcho_grid[lon_loc,lat_loc]+hcho[c,d]
                    hcho_cnts[lon_loc,lat_loc]=hcho_cnts[lon_loc,lat_loc]+1
                    
                    err_grid[lon_loc,lat_loc]=err_grid[lon_loc,lat_loc]+col_unc[c,d]
                    err_cnts[lon_loc,lat_loc]=err_cnts[lon_loc,lat_loc]+1
                    
                if cloud[c,d] <= 0.2 and flag[c,d] == 0 and hcho_rs[c,d] >= 0 and xtrack[c,d] == 0: 
                    ## find location on grid
                    lon_loc = np.where(abs(lon_centre - lons[c, d]) == abs(lon_centre - lons[c, d]).min())
                    lat_loc = np.where(abs(lat_centre - lats[c, d]) == abs(lat_centre - lats[c, d]).min())
                    
                    # in case of rare retrieval sites equally between points 1st location is taken
                    lon_loc = lon_loc[0]
                    lat_loc = lat_loc[0]
                    
                    ## store data on grid
                    hcho_grid_rs[lon_loc,lat_loc]=hcho_grid_rs[lon_loc,lat_loc]+hcho_rs[c,d]
                    hcho_cnts_rs[lon_loc,lat_loc]=hcho_cnts_rs[lon_loc,lat_loc]+1
                    
                    err_grid_rs[lon_loc,lat_loc]=err_grid_rs[lon_loc,lat_loc]+col_unc[c,d]
                    err_cnts_rs[lon_loc,lat_loc]=err_cnts_rs[lon_loc,lat_loc]+1
        
        good_data = np.where(hcho_cnts > 0)
        hcho_grid[good_data] = hcho_grid[good_data]/hcho_cnts[good_data]
        
        bad_data = np.where(hcho_cnts == 0)
        hcho_grid[bad_data] = mdi
        
        good_data = np.where(hcho_cnts > 0)
        err_grid[good_data] = err_grid[good_data]/err_cnts[good_data]
        
        bad_data = np.where(err_cnts == 0)
        err_grid[bad_data] = mdi
        
        good_data = np.where(hcho_cnts_rs > 0)
        hcho_grid_rs[good_data] = hcho_grid_rs[good_data]/hcho_cnts_rs[good_data]
        
        bad_data = np.where(hcho_cnts_rs == 0)
        hcho_grid_rs[bad_data] = mdi
        
        good_data = np.where(hcho_cnts_rs > 0)
        err_grid_rs[good_data] = err_grid_rs[good_data]/err_cnts_rs[good_data]
        
        bad_data = np.where(err_cnts_rs == 0)
        err_grid_rs[bad_data] = mdi
    
    ## store in large array
    hcho_grid_daily[i,:,:] = hcho_grid
    error_grid_daily[i,:,:] = err_grid
    
    hcho_grid_daily_rs[i,:,:] = hcho_grid_rs
    error_grid_daily_rs[i,:,:] = err_grid_rs
    



## save data to nc file
# falesafe to not overwrite data
try: ncfile.close()
except: pass

# creating new empty netCDF file
ncfile = Dataset('/home/users/egsands/sensecdt/users/egsands/outputs/hcho_grid_daily_2017.nc', mode='w', format='NETCDF4_CLASSIC')

# creating dimensions
lat_dim = ncfile.createDimension('lat', len(lat_centre))     # lat axis
lon_dim = ncfile.createDimension('lon', len(lon_centre))  # lon axis
time_dim = ncfile.createDimension('time', None)        # unlimited axis (enables appending)

# defining variables
# Define two variables with the same names as dimensions,
# a conventional way to define "coordinate variables".
lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'hours since 1800-01-01'
time.long_name = 'time'

# Define a 3D variables to hold the data
hcho = ncfile.createVariable('ColumnAmount',np.float64,('time','lon','lat')) # note: unlimited dimension is leftmost
hcho.units = 'molec/cm2' 
hcho.standard_name = 'Column Amount' 

error = ncfile.createVariable('ColumnUncertainty',np.float64,('time','lon','lat')) 
error.units = 'molec/cm2' 
error.standard_name = 'Column Uncertainty' 

hcho_rs = ncfile.createVariable('ReferenceSectorCorrectedVerticalColumn',np.float64,('time','lon','lat')) # note: unlimited dimension is leftmost
hcho_rs.units = 'molec/cm2' 
hcho_rs.standard_name = 'Reference Sector Corrected Vertical Column' 

error_rs = ncfile.createVariable('ColumnUncertaintyRS',np.float64,('time','lon','lat')) 
error_rs.units = 'molec/cm2' 
error_rs.standard_name = 'Column Uncertainty for Reference Sector Corrected Vertical Column Data' 


### adding data
hcho[:] = hcho_grid_daily[:]
error[:] = error_grid_daily[:]

hcho_rs[:] = hcho_grid_daily_rs[:]
error_rs[:] = error_grid_daily_rs[:]

# assigning data to time variable
dates = np.ones_like(date_array)
for i, j in enumerate(date_array):
    dates[i]= datetime.strptime(j, '%Y-%m-%d')
times = date2num(dates, time.units)
time[:] = times

# assigning data to coordinate variables
lat[:]=lat_centre
lon[:]=lon_centre


# checking final file
print(ncfile)

# close Dataset
ncfile.close()
print('Dataset is closed')

t2 = Time.perf_counter()
total_time = t2-t1

print(total_time)
