
import numpy as np
import pandas as pd
from modis_tools.auth import ModisSession
from modis_tools.resources import CollectionApi, GranuleApi
from modis_tools.granule_handler import GranuleHandler
import geopandas as gpd

start_date = '2020-01-01'
end_date = '2022-12-31'
output_path = '/mnt/locutus/remotesensing/r62/fire_danger/MODIS'
geojson_path = '/mnt/locutus/remotesensing/r62/fire_danger/California_State_Boundary/California_State_Boundary.geojson'
username = 'Russ_Earth_Data'
password = '-gEtT7Mv/e7W#Cg'

df = gpd.read_file(geojson_path)
bounds = df.geometry[0]

session = ModisSession(username=username, password=password)
collection_client = CollectionApi(session=session)
collections = collection_client.query(short_name="MOD13A1", version="061")
granule_client = GranuleApi.from_collection(collections[0], session=session)

granules_clipped = granule_client.query(start_date=start_date, end_date=end_date, bounding_box=bounds)
GranuleHandler.download_from_granules(granules_clipped, modis_session=session, path=output_path)


