from multiprocessing import Pool,Process,cpu_count,Manager
import time
import json,csv
import pandas as pd
from neuralNetwork import  neuralNetworks as nn,sigmoid,sigDeriv
import numpy as np
import equation
from treatments import conversion,getTargetVar
def taskManager(element,que,func):
    que.put(func(element))
print "starting"
start_time = time.time()
if __name__ == '__main__':
    raw=pd.read_json('input\\train.json')
    test=pd.read_json('input\\test.json')
    raw['target']=raw['interest_level'].map({'high':1,'medium':2,'low':3})

    print "dataset brought to python"
    # var,test=getLocation(test)
    # geoVar,raw=getLocation(raw)

    rawData=[]
    tester=[]
    raw1=raw.copy(deep=True)
    test1=test.copy(deep=True)
    print "extra copy created"
    raw1,var=conversion(raw1)
    test1,var=conversion(test1)
    print "starting subsetting dataset"
    for i in range(0,12):
        rawData.append(raw[raw1['loc'+str(i)]==1])
        tester.append(test[test1['loc'+str(i)]==1])
    noSeg=len(rawData)

#
# for i in range(0,len(geoVar)):




def nnOutput(element,analyse=False):
    print "starting to run for subset"+str(element)
    if analyse:analyseData=rawData[element].copy(deep=True)
    rawTransformed, var = conversion(rawData[element])
    rawTransformed=getTargetVar(rawTransformed)
    d=nn(listOfMatrix=[np.random.rand(len(var),6),np.random.rand(6,3)],input=rawTransformed[var].as_matrix(),output=rawTransformed[['high','medium','low']].as_matrix(),func=sigmoid,funcGradient=sigDeriv,iteration=100)
    d.findEstimates()
    prediction=d.predict(test=tester[element])
    if analyse:
        analyseData1 = analyseData.copy(deep=True)
        analyseData = getTargetVar(analyseData)
        temp = d.analyseObservation(dataSet=analyseData, var=var)
        temp.update(other=analyseData1,join='left')
        temp.to_csv('output\\cost'  + '.csv', sep=',')
    return prediction


if __name__ == '__main__':

    que=Manager().Queue()
    pool=Pool(processes=min(cpu_count(),noSeg-1))

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


# final=d.predict(test=test)
# final.to_csv('output\\test.csv', sep=',')
#ovr=equation.fit(raw,sheetName='sheet1',variables=var)
#equation.predict(ovr,test)

