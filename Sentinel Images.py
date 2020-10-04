import sentinelhub
import pandas as pd

# Input location of pre-processed dataset
file_path = 'data/wildfires.csv'
df = pd.read_csv(file_path)
df = df[df['FIRE_YEAR']>2013]
df = df[df['DISCOVERY_DOY']>60]
df = df.reset_index(drop=True)

from sentinelhub import SHConfig

# In case you put the credentials into the configuration file you can leave this unchanged

CLIENT_ID = '9a0eaca6-c18f-4fff-a73f-98722c7a080e'
CLIENT_SECRET = 'afGv}BMM8(^1y5Y^]w2Vz?9e%4#w.o5c49}V,HL9'

config = SHConfig()

if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

if config.sh_client_id == '' or config.sh_client_secret == '':
    print("Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret).")
#+- 0.005, +- 0.005 on each coordinate for boundaries


import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataSource, bbox_to_dimensions, DownloadRequest



def plot_image(image, factor=1.0, clip_range = None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])


#betsiboka_coords_wgs84 = [-120.99583, 40.02694, -121.01583, 40.04924]

def bbox_boundaries(dataframe):
    boundaries = []
    for i in range(len(dataframe)):
        long = dataframe['LONGITUDE'][i]
        lat = dataframe['LATITUDE'][i]
        long_left = long - 0.026
        long_right = long + 0.026
        lat_bottom = lat - 0.018
        lat_top = lat + 0.018
        bound_list = [round(long_left, 5), round(lat_bottom, 5), round(long_right, 5), round(lat_top, 5)]
        boundaries.append(bound_list)
    dataframe['BBOX'] = boundaries
    return dataframe

new_df = bbox_boundaries(df)


coords_wgs84 = new_df['BBOX'][0]
timeframe =

('2013-02-01', '2013-07-01')

resolution = 30
betsiboka_bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)
betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)

print(f'Image shape at {resolution} m resolution: {betsiboka_size} pixels')

evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

request_true_color = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_source=DataSource.LANDSAT8_L1C,
            time_interval=timeframe,
            mosaicking_order='leastCC'
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config
)

true_color_imgs = request_true_color.get_data()

print(f'Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}.')
print(f'Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}')

image = true_color_imgs[0]
print(f'Image type: {image.dtype}')

# plot function
# factor 1/255 to scale between 0-1
# factor 3.5 to increase brightness
plot_image(image, factor=3.5/255, clip_range=(0,1))

plt.show()


