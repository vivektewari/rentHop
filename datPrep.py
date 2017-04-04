from multiprocessing import Pool,Process,cpu_count,Manager
import time
import json,csv
import pandas as pd
from neuralNetwork import  neuralNetworks as nn,sigmoid,sigDeriv
import numpy as np
import equation
from treatments import conversion,getTargetVar
def taskManager(que,func,*args):
    que.put(func(*args))
print "starting"
start_time = time.time()
def segment(raw,raw1,test,test1,i):

        y=raw[raw1['loc' + str(i)] == 1],test[test1['loc' + str(i)] == 1]

        train=[]
        test=[]
        i=0
        med = y[0][y[0]['interest_level']=='high']['price'].quantile(0.9)
        output=train
        for element in y:

            if i==1 :output=test
            output.append(element[element['price'] > med])
            output.append(element[element['price'] <= med])
            i+=1
        print [train[0].shape[0],train[1].shape[0],test[0].shape[0],test[1].shape[0]]
        return train,test
extra=pd.DataFrame
def nnOutput(train,test,element,analyse=False):
    print "starting to run for subset"
    if analyse:analyseData=train.copy(deep=True)
    if element<>-1:rawTransformed, var = conversion(train)
    else:rawTransformed, var = conversion(train,first=False)
    rawTransformed=getTargetVar(rawTransformed)
    d=nn(listOfMatrix=[np.random.rand(len(var),8),np.random.rand(8,3)],input=rawTransformed[var].as_matrix(),output=rawTransformed[['high','medium','low']].as_matrix(),func=sigmoid,funcGradient=sigDeriv,iteration=500)
    d.findEstimates()
    print d.cost
    if analyse:
        analyseData1 = analyseData.copy(deep=True)
        analyseData = getTargetVar(analyseData)
        temp = d.analyseObservation(dataSet=analyseData, var=var)
        temp.update(other=analyseData1,join='left')
        temp.to_csv('output\\cost'  + '.csv', sep=',')
    if d.cost>0.6 and element<>-1:
        print element
        return element
    else:
        if element<>-1:prediction=d.predict(test=test)
        else :prediction=d.predict(test=test,first=False)

        return prediction

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
    noSeg=12
    pool = Pool(processes=min(cpu_count(), noSeg - 1))
    que = Manager().Queue()
    for i in range(0,noSeg):
        pool.apply_async(taskManager, args=( que,segment,raw,raw1,test,test1,i ))


    pool.close()
    pool.join()
    for i in range(0,noSeg):
        train,test=que.get()

        rawData+=train
        tester+=test

#
# for i in range(0,len(geoVar)):









    pool=Pool(processes=min(cpu_count(),noSeg-1))
    noSeg=len(rawData)
    for element in range(0,noSeg):
        pool.apply_async(taskManager, args=(que,nnOutput,rawData[element],tester[element],element))
    pool.close()
    pool.join()
    failures=[]
    pred =None
    for element in range(0,noSeg):
        got=que.get()
        if isinstance(got, int ):failures.append(got)
        elif pred is None :pred=got
        else:pred=pred.append(got)

    temp=rawData[failures[0]]
    tempt=tester[failures[0]]
    temp['location']=failures[0]
    tempt['location']=failures[0]
    fail=temp
    son=tempt



    for i in range(1,len(failures)):
        temp = rawData[failures[i]]
        tempt = tester[failures[i]]
        temp['location'] = failures[i]
        tempt['location'] = failures[i]
        fail=fail.append(temp)
        son=son.append(tempt)
    improvers=nnOutput(fail, son, -1)
    if pred is None:pred=improvers
    else:pred=pred.append(improvers)




    pred=pred.sort_index()
    print pred.shape[0]
    print np.sum(pred.as_matrix())
    pred.to_csv('output\\test.csv', sep=',')
    timeTaken = time.time() - start_time
    print "time taken is"
    print timeTaken


# final=d.predict(test=test)
# final.to_csv('output\\test.csv', sep=',')
#ovr=equation.fit(raw,sheetName='sheet1',variables=var)
#equation.predict(ovr,test)

