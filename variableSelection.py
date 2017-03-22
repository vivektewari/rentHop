import pandas as pd
import numpy as np
from datPrep import raw
def IV(dataset,variable,target,bins=1): #column:identifier,variable of interest,1/0 for target
    df0=dataset[list([variable])+list(target)]
    df1=df0
    # df1=(pd.qcut(df0[variable],bins)).to_frame('t')
    writer = pd.ExcelWriter('output\\bin.xlsx', engine='xlsxwriter')
    # df1.to_excel(writer, 'sheet1')
    g=df0.groupby([variable])['high','medium','low'].sum()
    # df1['mediumPerc'] = df0.groupby([variable])['medium'].mean()
    # df1['lowPerc'] = df0.groupby([variable])['low'].mean()
    print g
    #g.to_excel(writer, 'sheet1')


data=raw
data['high']=data.interest_level.map(lambda row:int(row=='high'))
data['medium']=data.interest_level.map(lambda row:int(row=='medium'))
data['low']=data.interest_level.map(lambda row:int(row=='low'))

IV(dataset=data,variable='FURNISHED',target=('high','medium','low'))

