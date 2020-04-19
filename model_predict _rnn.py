# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:29:55 2019

@author: Michael
"""

# Random Forest Classifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, asin, sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

class DBConnection:
    import mysql.connector

    mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    passwd = "password",
    database = "kinetic_location_db"

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

print(X.shape)
print(y.shape)

#standardizing the input feature
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Build model
model = Sequential()
model.add(Dense(100, input_shape=(5,), activation='relu'))
model.add(Dense(50, activation="relu"))
model.add(Dense(1, activation='relu'))

# Compile the model\
model.compile(loss='mean_squared_error', optimizer= "Adam", metrics=['mean_absolute_error'])

# Fit the model
es = EarlyStopping(monitor='val_loss', mode='min',patience=200, verbose=1)
model.fit(X_train,y_train,validation_data=(X_test, y_test),verbose=True,callbacks=[es],
          epochs=800,batch_size=32)

# Evaluate the model on new data
print(model.evaluate(X_test,y_test, verbose=False))

# Save the model and architechure to single file
model.save("model.h5")
print("Saved model to disk")

