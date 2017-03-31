import json,csv
import pandas as pd
from neuralNetwork import  neuralNetworks as nn,sigmoid,sigDeriv
import numpy as np
import equation
from treatments import conversion
raw=pd.read_json('input\\train.json')
test=pd.read_json('input\\test.json')
raw['target']=raw['interest_level'].map({'high':1,'medium':2,'low':3})
c=set()
count=1
# mergedFeatures=['Concierge','dishwasher','Laundry In Unit','elevator','Laundry In Building','Laundry in Building','HARDWOOD','Parking Space',\
#                   'Laundry Room','High Ceiling','Gym/Fitness','LAUNDRY','High Ceilings','Hardwood']



# writer = pd.ExcelWriter('output\\prepData2.xlsx', engine='xlsxwriter')
# raw.to_excel(writer, 'sheet1')
# writer.close()


priceSeg=[0,1500,2000,3000,4000,np.inf]
rawData=[]
tester=[]
for i in range(1,len(priceSeg)):
    rawData.append(raw[(raw['price']>=priceSeg[i-1]) & (raw['price']<priceSeg[i])])
    tester.append(test[(test['price']>=priceSeg[i-1]) & (test['price']<priceSeg[i])])

prediction=[]
for element in range(0,len(priceSeg)-1):
    rawTransformed, var = conversion(rawData[element])
    rawTransformed['high'] = rawData[element].interest_level.map(lambda row: int(row == 'high'))
    rawTransformed['medium'] = rawData[element].interest_level.map(lambda row: int(row == 'medium'))
    rawTransformed['low'] = rawData[element].interest_level.map(lambda row: int(row == 'low'))
    d=nn(listOfMatrix=[np.random.rand(len(var),10),np.random.rand(10,3)],input=rawTransformed[var].as_matrix(),output=rawTransformed[['high','medium','low']].as_matrix(),func=sigmoid,funcGradient=sigDeriv,iteration=500)
    d.findEstimates()
    if element==0:prediction=d.predict(test=tester[element])
    else:prediction=prediction.append(d.predict(test=tester[element]))
    temp = d.analyseObservation(dataSet=rawData[element], var=var)
    temp.to_csv('output\\cost' + str(priceSeg[element]) + '.csv', sep=',')
pred=prediction.sort_index()
pred.to_csv('output\\test.csv', sep=',')

#d.analyseObservation(dataSet=raw,var=var)
# final=d.predict(test=test)
# final.to_csv('output\\test.csv', sep=',')
#ovr=equation.fit(raw,sheetName='sheet1',variables=var)
#equation.predict(ovr,test)

