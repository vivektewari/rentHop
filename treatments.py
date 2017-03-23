import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
def conversion(raw):
    extraFeatures = ['bathrooms', 'bedrooms', 'picCount', 'price','newness']
    finalFeatures = ['Doorman', 'Furnished', 'Balcony', 'Renovated', 'No Fee', 'Elevator', 'Fitness Center', 'prewar',
                     'On-site Laundry', \
                     'HIGH CEILINGS', 'On-site Garage', 'LOWRISE', 'Cats Allowed', 'Roof-deck', 'Pool',
                     'Laundry in Unit', 'Pre-War', \
                     'Hardwood Floors', 'Swimming Pool', 'Fireplace', 'Multi-Level', 'Pets on approval',
                     'Granite Kitchen', 'Terrace', \
                     'LIVE IN SUPER', 'Stainless Steel Appliances', 'Loft', 'Garage', 'Dining Room',
                     'Wheelchair Access', 'SIMPLEX', \
                     'Dogs Allowed', 'Light', 'Washer/Dryer', 'Prewar', 'Reduced Fee', 'PublicOutdoor',
                     'Washer in Unit', 'Dryer in Unit', \
                     'High Speed Internet', 'Private Outdoor Space', 'Dishwasher', 'Green Building',
                     'Walk in Closet(s)', 'Exclusive', \
                     'Garden/Patio', 'Common Outdoor Space', 'Storage', 'New Construction', 'Roof Deck', 'Marble Bath',
                     'Live In Super', 'Outdoor Space']
    merger = {'Concierge': 'Doorman', 'dishwasher': 'Dishwasher', 'Laundry In Unit': 'Laundry in Unit',
              'elevator': 'Elevator', 'Laundry In Building': 'On-site Laundry', \
              'Laundry in Building': 'On-site Laundry', 'HARDWOOD': 'Hardwood Floors', 'Parking Space': 'Garage',
              'Laundry Room': 'Laundry in Building', \
              'High Ceiling': 'HIGH CEILINGS', 'Gym/Fitness': 'Fitness Center', 'LAUNDRY': 'Laundry in Building',
              'High Ceilings': 'HIGH CEILINGS', \
              'Hardwood': 'Hardwood Floors', 'Newly renovated': 'Renovated', 'On-site laundry': 'On-site Laundry'}

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
    t=pd.DataFrame(r,columns=pc,index=raw.index)
    raw=raw.join(t)

    return raw,extraFeatures +pc

