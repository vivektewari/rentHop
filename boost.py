from multiprocessing import Pool,Process,cpu_count,Manager
import time
import json,csv
import pandas as pd
from neuralNetwork import  neuralNetworks as nn
import numpy as np
import equation
from Booster import booster
from mathFunctions import sigmoid,sigDeriv
from treatments import conversion,getTargetVar
def taskManager(que,func,*args):
    que.put(func(*args))
print "starting"

start_time = time.time()
def segment(raw,raw1,test,test1,i):

        y=raw[raw1['loc' + str(i)] == 1],test[test1['loc' + str(i)] == 1]

        train=[y[0]]
        test=[y[1]]

        # med = y[0][y[0]['interest_level']=='high']['price'].quantile(0.9)
        # output=train
        # for element in y:
        #
        #     if i==1 :output=test
        #     output.append(element[element['price'] > med])
        #     output.append(element[element['price'] <= med])
        #     i+=1
        print [train[0].shape[0],test[0].shape[0]]
        return train,test

def nnOutput(train1,test,element,var,analyse=False):
    print "starting to run for subset"
    train=train1.copy(deep=True)
    if analyse:analyseData=train.copy(deep=True)
    rawTransformed=getTargetVar(train)
    nnObject=nn(listOfMatrix=[np.random.rand(len(var),8),np.random.rand(8,3)],input=rawTransformed[var].as_matrix(),output=rawTransformed[['high','medium','low']].as_matrix(),func=sigmoid,funcGradient=sigDeriv,variables=var,iteration=400)
    d=booster(classifier=nnObject,maxIteration=50,test=test,trainCopy=train1)
    if analyse:return d.weightSelecter(analyse)
    else:return d.nniterate()


if __name__ == '__main__':
    raw=pd.read_json('input\\train.json')
    test=pd.read_json('input\\test.json')
    raw['target']=raw['interest_level'].map({'high':1,'medium':2,'low':3})

    print "dataset brought to python"
    # var,test=getLocation(test)
    # geoVar,raw=getLocation(raw)

    rawData1=[]
    tester1=[]
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
        pool.apply_async(taskManager, args=( que,segment,raw,raw1,test,test1,i))


    pool.close()
    pool.join()
    for i in range(0,noSeg):
        train,test=que.get()

        rawData1+=train
        tester1+=test


    pool=Pool(processes=min(cpu_count(),noSeg-1))
    noSeg=len(rawData1)
    rawData=[]
    tester=[]
    for i in range(0, noSeg):
        rawData.append(conversion(rawData1[i])[0])
        tester.append(conversion(tester1[i])[0])

    for element in range(0,noSeg):
        pool.apply_async(taskManager, args=(que,nnOutput,rawData[element],tester[element],element,var))
    pool.close()
    pool.join()
    pred =None
    for element in range(0,noSeg):
        got=que.get()
        if pred is None :pred=got
        else:pred=pred.append(got)


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

