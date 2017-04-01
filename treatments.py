import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
def treatOutlier(raw):
    raw['price'].ix[raw['price'] > 13000] = 13000
    raw['price'].ix[raw['price'] < 1475] = 1475
    raw['latitude'].ix[raw['latitude'] < 40.6211755] = 40.6211755
    raw['latitude'].ix[raw['latitude'] > 40.8689] = 40.8689
    raw['longitude'].ix[raw['longitude'] < -74.018] = -74.018
    raw['longitude'].ix[raw['longitude'] > -73.8286755] = -73.8286755
    return raw



def getPCA(dataSet,variables,pcComponents):# adds pc components as pc0,pc1... for variables in the dataset
    feat=dataSet[variables]
    pca = PCA(n_components=pcComponents)
    pca.fit(feat)
    pc=['pc'+str(i) for i in range(0,pcComponents)]
    t=pd.DataFrame(np.dot(feat.as_matrix(),np.transpose(pca.components_)),columns=pc,index=dataSet.index)
    dataSet=dataSet.join(t)
    return pc,dataSet

def convFeatures(dataSet):
    dataSet=treatOutlier(dataSet)
    dataSet.describe()
    finalFeatures = ['NOFEE', 'HARDWOODFLOORS', 'DISHWASHER', 'ON-SITELAUNDRY', 'OUTDOORSPACE']
    merger = {'dishwasher': 'Dishwasher', 'Laundry In Building': 'On-site Laundry', \
              'Laundry in Building': 'On-site Laundry', 'HARDWOOD': 'Hardwood Floors', 'Hardwood': 'Hardwood Floors', \
              'On-site laundry': 'On-site Laundry'}

    cleanFeatures = []
    cleanMerger = {}
    for element in finalFeatures:
        cleanFeatures.append((element.upper()).replace(" ", ""))
        dataSet[(element.upper()).replace(" ", "")] = 0
    for element in merger.keys(): cleanMerger[(element.upper()).replace(" ", "")] = ((merger[element]).upper()).replace(
        " ", "")
    dataSet['picCount'] = 0
    for index, row in dataSet.iterrows():
        d = set(row['features'])
        dataSet.set_value(index, 'picCount', len(row['photos']))
        for element in d:
            cleanElement = (element.upper()).replace(" ", "")
            if cleanElement in cleanFeatures:
                dataSet.set_value(index, cleanElement, 1)
            elif cleanElement in cleanMerger.keys():
                dataSet.set_value(index, cleanMerger[cleanElement], 1)
    return cleanFeatures,dataSet
def getLocation(raw):
    latMax=40.8689
    latMin=40.6211755
    lonMax=-73.8286755
    lonMin=-74.018

    latInt=(latMax-latMin)/4
    lonInt=(lonMax-lonMin)/4
    latDivision=[latMin+i*latInt for i in range(0,5)]
    lonDivision=[lonMin+i*lonInt for i in range(0,5)]
    geoVar=[]
    for i in range(1,5):
        for j in range(1, 5):
            raw['geo_'+str(i*10+j*1)]=raw.apply(lambda row: int(row['latitude']<latDivision[i] and row['latitude']>=latDivision[i-1] and row['longitude']<lonDivision[i] and row['longitude']>=lonDivision[i-1]),axis=1)
            geoVar.append('geo_'+str(i*10+j*1))
    return geoVar,raw
def conversion(raw):
    raw=treatOutlier(raw)
    cleanFeatures,raw=convFeatures(raw)
    extraFeatures = ['bathrooms', 'bedrooms', 'picCount', 'price','newness']
    raw['newness']= (pd.to_datetime('2005/11/23')-pd.to_datetime(raw['created'])).dt.days
    geoVar,raw=getLocation(raw)
    standarize=cleanFeatures+extraFeatures
    raw_normalize = (raw[standarize] - raw[standarize].mean()) / (raw[standarize].std())
    raw['intercept'] = 1
    raw.update(raw_normalize)
    var=standarize+geoVar+['intercept']

    #34.0126,44.8835

    return raw,var

#0.922148515716
# 0.944995418225
# 0.744928707697
# 0.609586396456
# 0.514380318224