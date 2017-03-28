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
    raw['intercept']=1
    t=pd.DataFrame(r,columns=pc,index=raw.index)
    raw=raw.join(t)
    return raw,finalFeatures+['intercept']

