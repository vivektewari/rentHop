import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
def mergeLocation(dataSet,nameList):
    dict={0:[0],1:[1,2],2:[5,9,10,11,3,4],3:[7,8],4:[12,13],5:[14,15],6:[17,18,22,23,24,25,28,30,31,16,17,29,35],7:[19,20],8:[21],9:[26,27],10:[32,33,34],11:[6]}
    varList=[]
    for key in dict.keys():
        cols=[]
        for ele in dict[key]:cols.append(nameList[ele])
        dataSet['loc'+str(key)]=pd.DataFrame(np.sum(dataSet[cols].as_matrix(),axis=1),columns=['var'],index=dataSet.index)['var']
        varList.append('loc'+str(key))
    return varList,dataSet

def getTargetVar(data):
    for i in ['high','medium','low']:
        data[i]=data.interest_level.map(lambda row: int(row == i))
    return data
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
    latMax=40.8690#keeping .0000001 more to cover all the data point
    latMin=40.6211755
    lonMax=-73.8286754#keeping .0000001 more to cover all the data point
    lonMin=-74.018
    noLat=5
    noLon=3

    latInt=(latMax-latMin)/noLat
    lonInt=(lonMax-lonMin)/noLon
    latDivision=np.array([0*latInt,2*latInt,2.5*latInt,3*latInt,3.5*latInt,4*latInt,5*latInt])+latMin
    lonDivision=np.array([0*lonInt, 0.33*lonInt,0.66*lonInt,1*lonInt,1.33*lonInt,1.66*lonInt,3*lonInt])+lonMin
    geoVar=[]
    for i in range(1,len(latDivision)):
        for j in range(1, len(lonDivision)):
            raw['geo_'+str(i*10+j*1)]=raw.apply(lambda row: int(row['latitude']<latDivision[i] and row['latitude']>=latDivision[i-1] and row['longitude']<lonDivision[j] and row['longitude']>=lonDivision[j-1]),axis=1)
            geoVar.append('geo_'+str(i*10+j*1))
    return geoVar,raw
def conversion(raw):
    raw=treatOutlier(raw)
    cleanFeatures,raw=convFeatures(raw)
    extraFeatures = ['bathrooms', 'bedrooms', 'picCount', 'price','newness']
    raw['newness']= (pd.to_datetime('2005/11/23')-pd.to_datetime(raw['created'])).dt.days
    geoVar,raw=getLocation(raw)
    geo,raw=mergeLocation(raw,geoVar)
    standarize=cleanFeatures+extraFeatures
    raw_normalize = (raw[standarize] - raw[standarize].mean()) / (raw[standarize].std())
    raw['intercept'] = 1
    raw.update(raw_normalize)
    var=standarize+['intercept']


    #34.0126,44.8835


    return raw,var

#0.922148515716
# 0.944995418225
# 0.744928707697
# 0.609586396456
# 0.514380318224