#!/usr/bin/python3
# conda activate r62GEDI

import geopandas as gpd
import requests
import datetime as dt 
from pydap.client import open_url
import xarray as xr
import os
import sys
sys.path.append('/path/to/functions/')

from STANDARD_FUNCTIONS import runcmd

#########################################################################################################################

data_path = 'path/to/where/you/are/storing/project/data'

geojson_path = f'{data_path}/California_State_Boundary/California_State_Boundary.geojson'
output_path = f'{data_path}/daymet'

start_date = dt.datetime(2020, 1, 1)
end_date = dt.datetime(2023, 12, 31)

VARS = ['prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'] # omitting dayl
convert_to_tif = True

######################################################################################################

os.makedirs(f'{data_path}/California_State_Boundary/', exist_ok=True)
os.makedirs(f'{data_path}/daymet/', exist_ok=True)

ca_geojson_4326 = gpd.read_file(geojson_path)
ca_4326_bounds = ca_geojson_4326.bounds

# Reprojecting to LCC
daymet_proj = "+proj=lcc +ellps=WGS84 +a=6378137 +b=6356752.314245 +lat_1=25 +lat_2=60 +lon_0=-100 +lat_0=42.5 +x_0=0 +y_0=0 +units=m +no_defs"
ca_gpd_lcc = ca_geojson_4326.to_crs(daymet_proj)

dt_format = '%Y-%m-%dT%H:%M:%SZ' # format requirement for datetime search
temporal_str = start_date.strftime(dt_format) + ',' + end_date.strftime(dt_format)

# Searching for granules

daymet_doi = '10.3334/ORNLDAAC/2129' # define the Daymet V4 Daily Data DOI as the variable `daymet_doi`
cmrurl='https://cmr.earthdata.nasa.gov/search/' # define the base url of NASA's CMR API as the variable `cmrurl`
doisearch = cmrurl + 'collections.json?doi=' + daymet_doi # Create the Earthdata Collections URL
print('Earthdata Collections URL: Daymet V4 Daily -->', doisearch)

# From the doisearch, we can obtain the ConceptID for the Daymet V4 Daily data
# We'll search the json response of the Daymet metadata for "id" within the 'entry' dictionary key
response = requests.get(doisearch)
collection = response.json()['feed']['entry'][0] 
concept_id = collection['id']
print('NASA Earthdata Concept_ID --> ' , concept_id)

granulesearch = cmrurl + 'granules.json?collection_concept_id=' + concept_id + \
                '&page_size=1000' + '&temporal=' + temporal_str + \
                '&bounding_box[]=' + ','.join([str(e) for e in ca_4326_bounds.loc[0]])

response = requests.get(granulesearch)
granules = response.json()['feed']['entry']
granule_names = []  # create an empty array
print('*'*20)

for g in granules:
    granule_name = g['title'] # fill the array with granule names that match our search parameters
    if any(variable in granule_name for variable in VARS):
        granule_names.append(granule_name)
        print(granule_name)
		
print('*'*20)
print('THERE WERE:', len(granule_names), 'GRANULES FOUND')
print('*'*20)

# see data url at this link to see how the granule urls were constructed: 
# https://thredds.daac.ornl.gov/thredds/dodsC/ornldaac/2129/daymet_v4_daily_na_srad_1980.nc.html

for g_name in granule_names:
    g_name = g_name.replace('Daymet_Daily_V4R1.', '')
    if os.path.exists(f'{output_path}/{g_name}'):
        print('FILE:', g_name, 'ALREADY IMPORTED')
    else:
        print('IMPORTING GRANULE:', g_name)
        
        granule_dap = f'https://thredds.daac.ornl.gov/thredds/dodsC/ornldaac/2129/{g_name}'
        variable = g_name.split('_')[-2]

        # Using pydap's open_url 
        thredds_ds = open_url(granule_dap)
        ds = xr.open_dataset(xr.backends.PydapDataStore(thredds_ds), decode_coords="all")
        ds = ds[variable].sel(x=slice(ca_gpd_lcc.bounds.minx[0],
                                      ca_gpd_lcc.bounds.maxx[0]),
                                      y=slice(ca_gpd_lcc.bounds.maxy[0],
                                      ca_gpd_lcc.bounds.miny[0]))

        ds.to_netcdf(f'{output_path}/{g_name}')

if convert_to_tif:
    os.chdir(output_path)
    daymet_files = [i for i in os.listdir() if i.startswith('daymet_v4_daily') and i.endswith('.nc')]

    for daymet_file in daymet_files:
        new_name = daymet_file[:-2] + 'tif'
        runcmd(f'gdal_translate -of GTiff NETCDF:"{daymet_file}" {new_name}')
