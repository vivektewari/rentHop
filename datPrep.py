import json,csv
import pandas as pd
from neuralNetwork import  neuralNetworks as nn,sigmoid,sigDeriv
import numpy as np
import equation
from treatments import conversion
raw=pd.read_json('input\\train.json')[0:1000]
#test=pd.read_json('input\\test.json')
raw['target']=raw['interest_level'].map({'high':1,'medium':2,'low':3})
c=set()
count=1
# mergedFeatures=['Concierge','dishwasher','Laundry In Unit','elevator','Laundry In Building','Laundry in Building','HARDWOOD','Parking Space',\
#                   'Laundry Room','High Ceiling','Gym/Fitness','LAUNDRY','High Ceilings','Hardwood']



# writer = pd.ExcelWriter('output\\prepData2.xlsx', engine='xlsxwriter')
# raw.to_excel(writer, 'sheet1')
# writer.close()
raw,var=conversion(raw)
raw['high']=raw.interest_level.map(lambda row:int(row=='high'))
raw['medium']=raw.interest_level.map(lambda row:int(row=='medium'))
raw['low']=raw.interest_level.map(lambda row:int(row=='low'))
nn.neuralNetworks(input=raw[[var]].as_matrix(),output=raw[['high','nedium','low']].as_matrix(),func=sigmoid,funcGradient=sigDeriv)
#ovr=equation.fit(raw,sheetName='sheet1',variables=var)
#equation.predict(ovr,test)

