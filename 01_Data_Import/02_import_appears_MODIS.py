#!/usr/bin/python3
# conda activate r62GEDI

import requests as r
import getpass, pprint, time, os, cgi, json
import geopandas as gpd

# https://lpdaac.usgs.gov/resources/e-learning/getting-started-with-the-a%CF%81%CF%81eears-api-submitting-and-downloading-an-area-request/

api = 'https://appeears.earthdatacloud.nasa.gov/api/'

user = getpass.getpass(prompt = 'Enter username (Russ_Earth_Data)')
password = getpass.getpass(prompt = 'Enter password: (Rockytop0153!)')
token_response = r.post(f'{api}login', auth=(user, password)).json()  # Insert API URL, call login service, provide credentials & return json

shp_file_path = '/mnt/locutus/remotesensing/r62/fire_danger/California_State_Boundary/California_State_Boundary.shp'
task_name = input('Enter a Task Name: ')
output_dir = '/mnt/locutus/remotesensing/r62/fire_danger'

# SETTING UP MY TOKEN:
del user, password
token = token_response['token']
head = {'Authorization': 'Bearer {}'.format(token)} 

# PULLING IN MY SHAPE FILE
nps_gc = gpd.read_file(f'{shp_file_path}').to_json()
nps_gc = json.loads(nps_gc)  

product_response = r.get('{}product'.format(api)).json()
print('AρρEEARS currently supports {} products.'.format(len(product_response)))  

products = {p['ProductAndVersion']: p for p in product_response} # Create a dictionary #indexed by product name & version

prodNames = {p['ProductAndVersion'] for p in product_response}
# Make list of all products (including version)

# SEARCH FOR THE PRODUCT YOU ARE LOOKING FOR:
#for p in prodNames:                                            #
#Make for loop to search list of products 'Description' for a keyword                
    #if ('NDVI' in products[p]['Description']) and ('250m' in products[p]['Resolution']):
    # pprint.pprint(products[p])

prods = ['MOD13Q1.061']
layers = [(prods[0],'_250m_16_days_NDVI')]
ndvi_response = r.get('{}product/{}'.format(api, prods[0])).json()
prodLayer = []
for l in layers:
    prodLayer.append({
            "layer": l[1],
            "product": l[0]
          })
print(prodLayer)

# GET AVAILABLE PROJECTIONS FROM APPEARS
# Create a dictionary of projections with projection Name as the keys
projections = r.get('{}spatial/proj'.format(api)).json()
projs = {}
for p in projections:
    projs[p['Name']] = p 
print(list(projs.keys()))

task_type = ['point','area']        # Type of task, area or point
proj = projs['geographic']['Name']  # Set output projection 
outFormat = ['geotiff']  # Set output file format type
startDate = '01-01'            # Start of the date range 
endDate = '12-31'              # End of the date range
recurring = True                   # Specify True for a recurring date
yearRange = [2020, 2023]

task = {
    'task_type': task_type[1],
    'task_name': task_name,
    'params': {
         'dates': [
         {
             'startDate': startDate,
             'endDate': endDate,
             'recurring': recurring,
             'yearRange': yearRange  
         }],
         'layers': prodLayer,
         'output': {
                 'format': {
                         'type': outFormat[0]}, 
                         'projection': proj},
         'geo': nps_gc,
    }
}

task_response = r.post('{}task'.format(api), json=task, headers=head).json()
print(task_response)

params = {'limit': 2, 'pretty': True} # Limit API response to 2 most recent entries, return as pretty json
tasks_response = r.get('{}task'.format(api), params=params, headers=head).json() 
task_id = task_response['task_id'] # Set task id from request submission
status_response = r.get('{}status/{}'.format(api, task_id), headers=head).json() 


starttime = time.time()
while r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'] != 'done':
    print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])
    time.sleep(20.0 - ((time.time() - starttime) % 20.0))
print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])

destDir = os.path.join(output_dir, task_name) 
if not os.path.exists(destDir):
	os.makedirs(destDir)
	
bundle = r.get('{}bundle/{}'.format(api,task_id), headers=head).json()
files = {}                                                       # Create empty dictionary
for f in bundle['files']:
	files[f['file_id']] = f['file_name'] 

for f in files:
    dl = r.get('{}bundle/{}/{}'.format(api, task_id, f), headers=head, stream=True, allow_redirects = 'True') # Get a stream to the bundle file
    if files[f].endswith('.tif'):
        filename = files[f].split('/')[1]
    else:
        filename = files[f] 
    filepath = os.path.join(destDir, filename) # Create output file path
    with open(filepath, 'wb') as f: # Write file to dest dir
        for data in dl.iter_content(chunk_size=8192): f.write(data) 
print(f'Downloaded files can be found at: {destDir}')
