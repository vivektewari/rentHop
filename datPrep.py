import json,csv
import pandas as pd
import numpy as np
import equation
from treatments import conversion
raw=pd.read_json('input\\train.json')[0:1000]
test=pd.read_json('input\\test.json')
raw['target']=raw['interest_level'].map({'high':1,'medium':2,'low':3})
c=set()
count=1
# mergedFeatures=['Concierge','dishwasher','Laundry In Unit','elevator','Laundry In Building','Laundry in Building','HARDWOOD','Parking Space',\
#                   'Laundry Room','High Ceiling','Gym/Fitness','LAUNDRY','High Ceilings','Hardwood']



# writer = pd.ExcelWriter('output\\prepData2.xlsx', engine='xlsxwriter')
# raw.to_excel(writer, 'sheet1')
# writer.close()
raw,var=conversion(raw)
#ovr=equation.fit(raw,sheetName='sheet1',variables=var)
#equation.predict(ovr,test)

