
def conversion(raw):
    extraFeatures = ['bathrooms', 'bedrooms', 'picCount', 'price']

    finalFeatures = ['FURNISHED', 'BALCONY', 'RENOVATED', 'NOFEE', 'ON-SITELAUNDRY', 'HIGHCEILINGS', 'LAUNDRYINUNIT',
                     'HARDWOODFLOORS', 'MULTI-LEVEL', \
                     'PETSONAPPROVAL', 'GRANITEKITCHEN', 'TERRACE', 'STAINLESSSTEELAPPLIANCES', 'LOFT', 'DININGROOM',
                     'WHEELCHAIRACCESS', 'LIGHT', 'REDUCEDFEE', \
                     'HIGHSPEEDINTERNET', 'PRIVATEOUTDOORSPACE', 'DISHWASHER', 'WALKINCLOSET(S)', 'EXCLUSIVE',
                     'GARDEN/PATIO', 'COMMONOUTDOORSPACE', \
                     'NEWCONSTRUCTION', 'ROOFDECK', 'MARBLEBATH', 'OUTDOORSPACE']
    merger = {'dishwasher': 'Dishwasher', 'Laundry In Unit': 'Laundry in Unit',
              'Laundry In Building': 'On-site Laundry', \
              'Laundry in Building': 'On-site Laundry', 'HARDWOOD': 'Hardwood Floors',
              'Laundry Room': 'On-site Laundry', \
              'High Ceiling': 'HIGH CEILINGS', 'LAUNDRY': 'On-site Laundry', 'High Ceilings': 'HIGH CEILINGS', \
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
    return raw,extraFeatures+cleanFeatures

