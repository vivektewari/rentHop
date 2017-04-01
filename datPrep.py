from multiprocessing import Pool,Process,cpu_count,Manager
import time
import json,csv
import pandas as pd
from neuralNetwork import  neuralNetworks as nn,sigmoid,sigDeriv
import numpy as np
import equation
from treatments import conversion,getLocation
def taskManager(element,que,func):
    que.put(func(element))

raw=pd.read_json('input\\train.json')
test=pd.read_json('input\\test.json')
raw['target']=raw['interest_level'].map({'high':1,'medium':2,'low':3})


# var,test=getLocation(test)
# geoVar,raw=getLocation(raw)
rawData=[]
tester=[]
noSeg=2#len(geoVar)
#
# for i in range(0,len(geoVar)):
tester.append(test)


def nnOutput(element):
    rawTransformed, var = conversion(rawData[element])
    rawTransformed['high'] = rawData[element].interest_level.map(lambda row: int(row == 'high'))
    rawTransformed['medium'] = rawData[element].interest_level.map(lambda row: int(row == 'medium'))
    rawTransformed['low'] = rawData[element].interest_level.map(lambda row: int(row == 'low'))
    d=nn(listOfMatrix=[np.random.rand(len(var),10),np.random.rand(10,3)],input=rawTransformed[var].as_matrix(),output=rawTransformed[['high','medium','low']].as_matrix(),func=sigmoid,funcGradient=sigDeriv,iteration=500)
    d.findEstimates()
    prediction=d.predict(test=tester[element])
    temp = d.analyseObservation(dataSet=rawData[element], var=var)
    temp.to_csv('output\\cost'  + '.csv', sep=',')
    return prediction


if __name__ == '__main__':
    start_time=time.time()
    que=Manager().Queue()
    pool=Pool(processes=cpu_count())
    prediction=[i for i in range(0,noSeg-1)]

    for element in range(0,noSeg-1):
        pool.apply_async(taskManager, args=(element,que,nnOutput))
    pool.close()
    pool.join()
    for element in range(0,noSeg-1):
        if element==0 :pred=que.get()
        else:pred=pred.append(que.get())
    pred=pred.sort_index()
    pred.to_csv('output\\test.csv', sep=',')
    timeTaken = time.time() - start_time
    print "time taken is"
    print timeTaken

#d.analyseObservation(dataSet=raw,var=var)
# final=d.predict(test=test)
# final.to_csv('output\\test.csv', sep=',')
#ovr=equation.fit(raw,sheetName='sheet1',variables=var)
#equation.predict(ovr,test)

