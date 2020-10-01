import pandas as pd
import numpy as np
import time
import datetime



file_path = 'data/wildfires.csv'
df = pd.read_csv(file_path)


# Working on minutes elapsed feature
import time

new_times = []
for row in df['DISCOVERY_TIME']:
    new = str(row)
    new_times.append(new)


df['DISCOVERY_TIME'] = new_times

new_times = []
i=0
for row in df['DISCOVERY_TIME']:
    new = time.strptime(row, "%H%M")
    new_times.append(new)
    print(i)
    i=i+1

df['disc_time'] = new_times