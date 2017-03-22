import pandas as pd
import numpy as np
from datPrep import raw,var
def IV(dataset,variable,target,bins=1): #column:identifier,variable of interest,1/0 for target
    df0=dataset[list([variable])+list(target)]


    g=df0.groupby([variable])['high','medium','low'].sum()/df0.sum()

    score=(g.diff(periods=1,axis=1)**2)[-2:].sum(axis=1)
    if g['medium'][0]*g['low'][0]<0 :return 0
    else :return score[0]



data=raw
data['high']=data.interest_level.map(lambda row:int(row=='high'))
data['medium']=data.interest_level.map(lambda row:int(row=='medium'))
data['low']=data.interest_level.map(lambda row:int(row=='low'))
features=[x for x in var if x not in ['bathrooms', 'bedrooms', 'picCount', 'price']]
dict={}
for element in features:
    dict[element]=IV(dataset=data,variable=element,target=('high','medium','low'))
df=pd.DataFrame(dict.items(),columns=['variable','iv'])
writer = pd.ExcelWriter('output\\IV.xlsx', engine='xlsxwriter')
df.to_excel(writer, 'sheet1')
writer.close()

