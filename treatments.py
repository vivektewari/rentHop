import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
def conversion(raw):
    extraFeatures = ['bathrooms', 'bedrooms', 'picCount', 'price','newness']
    finalFeatures = ['NOFEE', 'HARDWOODFLOORS', 'DISHWASHER', 'ON-SITELAUNDRY', 'OUTDOORSPACE']
    merger = {'dishwasher': 'Dishwasher', 'Laundry In Building': 'On-site Laundry', \
                               'Laundry in Building': 'On-site Laundry', 'HARDWOOD': 'Hardwood Floors',
                               'Hardwood': 'Hardwood Floors', \
                               'On-site laundry': 'On-site Laundry'}

    cleanFeatures = []
    cleanMerger = {}
    for element in finalFeatures:
        cleanFeatures.append((element.upper()).replace(" ", ""))
        raw[(element.upper()).replace(" ", "")] = 0
    for element in merger.keys(): cleanMerger[(element.upper()).replace(" ", "")] = ((merger[element]).upper()).replace(
        " ", "")
    raw['picCount'] = 0
    for index, row in raw.iterrows():
        d = set(row['features'])
        raw.set_value(index, 'picCount', len(row['photos']))
        for element in d:
            cleanElement = (element.upper()).replace(" ", "")
            if cleanElement in cleanFeatures:
                raw.set_value(index, cleanElement, 1)
            elif cleanElement in cleanMerger.keys():
                raw.set_value(index, cleanMerger[cleanElement], 1)
    raw['newness']= (pd.to_datetime('2005/11/23')-pd.to_datetime(raw['created'])).dt.days
    feat=raw[cleanFeatures]
    pca = PCA(n_components=5)
    pca.fit(feat)
    df=pd.DataFrame(np.transpose(pca.components_))

    r=np.dot(feat,df)
    pc=['pc1','pc2','pc3','pc4','pc5']
    #t=pd.DataFrame(r,columns=pc,index=raw.index)
    #raw=raw.join(t)
    #raw['latL']=pd.qcut(raw['latitude'],4,labels=False)
    #raw['longL']= pd.qcut(raw['longitude'],4,labels=False)
    #raw=raw.join(raw.groupby(by=['latL','longL'])['target'].mean(), on=['latL','longL'], rsuffix='_r')

    var=finalFeatures+extraFeatures
    raw_normalize = (raw[var] - raw[var].mean()) / (raw[var].std())
    raw_normalize['intercept'] = 1

    #34.0126,44.8835

    return raw_normalize,var+['intercept']

