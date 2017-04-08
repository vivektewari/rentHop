import pandas as pd
import numpy as np

def IV(rawset,variable,target,bins=1): #column:identifier,variable of interest,1/0 for target
    df0=rawset[list([variable])+list(target)]


    g=df0.groupby([variable])['high','medium','low'].sum()/df0.sum()

    score=(g.diff(periods=1,axis=1)**2)[-2:].sum(axis=1)
    if g['medium'][0]*g['low'][0]<0 :return 0
    else :return score[0]




raw = pd.read_json('input\\train.json')
test=pd.read_json('input\\test.json')
raw['high']=raw.interest_level.map(lambda row:int(row=='high'))
raw['medium']=raw.interest_level.map(lambda row:int(row=='medium'))
raw['low']=raw.interest_level.map(lambda row:int(row=='low'))

raw1=raw.groupby(['manager_id'])['high','medium','low'].sum()
raw1['manger_id']=raw1.index

test1=test.groupby(['manager_id'])['price','bathrooms'].count()
test1['manager_id']=test1.index

joined=test1.join(raw1,on=['manager_id'],how='left')

writer = pd.ExcelWriter('output\\manager.xlsx', engine='xlsxwriter')
joined.to_excel(writer, 'sheet1')
writer.close()

features=[x for x in var if x not in ['bathrooms', 'bedrooms', 'picCount', 'price']]
dict={}
for element in features:
    dict[element]=IV(rawset=raw,variable=element,target=('high','medium','low'))
df=pd.rawFrame(dict.items(),columns=['variable','iv'])
writer = pd.ExcelWriter('output\\IV.xlsx', engine='xlsxwriter')
df.to_excel(writer, 'sheet1')
writer.close()

