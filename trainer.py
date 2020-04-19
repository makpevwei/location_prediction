import numpy as np
#import pickle
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from math import radians, cos, sin, asin, sqrt
#import h5py
import json


class DBConnection:
    import mysql.connector

    mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    passwd = "0400",
    database = "location"

    )
    
def getTest():
    locations = []
    mydb = DBConnection().mydb
    mycursor = mydb.cursor()
    mycursor.execute("SELECT locPhoneImei,locLatitude, locLongitude, locAltitude,locTime FROM location where locPhoneImei ='354726072625546' order by locDate desc")
    myResult = mycursor.fetchall()
    for x in myResult:
        locations.append(x)
    return locations
                     
testLocations = getTest()

lat = []
lon = []
times = []
locPhoneImei = []
locAltitude = []
for loc in testLocations:
    locPhoneImei.append(loc[0])
    lat.append(loc[1])
    lon.append(loc[2])
    locAltitude.append(loc[3])
    times.append(loc[4])


df_test = pd.DataFrame(
    {'LocPhoneImei' : locPhoneImei,
     'lat' : lat,
     'lon' : lon,
      'locAltitude': locAltitude,
     'time' : times
    })

df_test['lat'] = df_test['lat'].astype(float)
df_test['lon'] = df_test['lon'].astype(float)

df_test.drop('LocPhoneImei', axis=1,inplace=True)
df_test = df_test.sort_values('time', ascending=False)

df_test['year'] = pd.DatetimeIndex(df_test['time']).year
df_test['month'] = pd.DatetimeIndex(df_test['time']).month
df_test['day'] = pd.DatetimeIndex(df_test['time']).day
df_test['hour'] = pd.DatetimeIndex(df_test['time']).hour
df_test['min'] = pd.DatetimeIndex(df_test['time']).minute
df_test['sec'] = pd.DatetimeIndex(df_test['time']).second

df_test['locAltitude'] = df_test['locAltitude'].astype(float)

def single_pt_haversine(lat, lng, degrees=True):
    """
    'Single-point' Haversine: Calculates the great circle distance
    between a point on Earth and the (0, 0) lat-long coordinate
    """
    r = 6371 # Earth's radius (km). Have r = 3956 if you want miles

    # Convert decimal degrees to radians
    if degrees:
        lat, lng = map(radians, [lat, lng])

    # 'Single-point' Haversine formula
    a = sin(lat/2)**2 + cos(lat) * sin(lng/2)**2
    d = 2 * r * asin(sqrt(a)) 

    return d

df_test['harvesine_distance'] = [single_pt_haversine(lat, lon) for lat, lon in zip(df_test.lat, df_test.lon)]


X = np.array(df_test.drop(['time','year','lat','lon','locAltitude','harvesine_distance'], axis=1))
y = np.array(df_test['harvesine_distance'])

# split into input (X) and output (Y) variables
print(X.shape)
print(y.shape)

'''     Now we can define different Keras Models that will be different 
        with respect to 'number of hidden layers', 'number of epochs' and
        'nuerons in the initial layer'.
'''
# define base model

# Build model
model = Sequential()
model.add(Dense(100, input_dim=5, activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation="relu"))
model.add(Dense(1, kernel_initializer='normal'))

# Compile model (this tells how the model is to be trained)
model.compile(loss='mean_squared_error', optimizer='adam')

print("FITTING THE MODEL NOW")
# fitting the model
model.fit(X,y,batch_size = 5 , epochs = 100)

# saving the model by serializing it using json (same thing can be done using YAML)
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

print("MODEL HAS BEEN SAVED SUCCESSFULLY")
# loaded model will only take in 2D array values
