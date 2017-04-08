from multiprocessing import Pool,Process,cpu_count,Manager
import time
import json,csv
import pandas as pd
from neuralNetwork import  neuralNetworks as nn
import numpy as np
import equation
from mathFunctions import sigmoid,sigDeriv
from treatments import conversion,getTargetVar,getManager
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

def nnOutput(train,test,element,var,analyse=False):
    print "starting to run for subset"
    if analyse:analyseData=train.copy(deep=True)
    rawTransformed=train
    d=nn(listOfMatrix=[np.random.rand(len(var),8),np.random.rand(8,3)],input=rawTransformed[var].as_matrix(),output=rawTransformed[['high','medium','low']].as_matrix(),func=sigmoid,funcGradient=sigDeriv,variables=var,iteration=0)
    d.findEstimates()
    print d.cost
    if analyse:
        analyseData1 = analyseData.copy(deep=True)
        temp = d.analyseObservation(dataSet=analyseData)
        temp=temp[var + ['cost'] + ['high', 'medium', 'low']]
        temp.update(other=analyseData1,join='left')
        temp.to_csv('output\\cost'  + '.csv', sep=',')
    prediction=d.predict(test=tester)

    return prediction

if __name__ == '__main__':
    raw=pd.read_json('input\\train.json')
    test=pd.read_json('input\\test.json')


    print "dataset brought to python"
    # var,test=getLocation(test)
    # geoVar,raw=getLocation(raw)

    # rawData1=[]
    # tester1=[]
    # raw1=raw.copy(deep=True)
    # test1=test.copy(deep=True)
    # print "extra copy created"
    # raw1,var=conversion(raw1)
    # test1,var=conversion(test1)
    # print "starting subsetting dataset"
    # noSeg=12
    # pool = Pool(processes=min(cpu_count(), noSeg - 1))
    # que = Manager().Queue()
    # for i in range(0,noSeg):
    #     pool.apply_async(taskManager, args=( que,segment,raw,raw1,test,test1,i))
    #
    #
    # pool.close()
    # pool.join()
    # for i in range(0,noSeg):
    #     train,test=que.get()
    #
    #     rawData1+=train
    #     tester1+=test
    #
    # pool=Pool(processes=min(cpu_count(),noSeg-1))
    # noSeg=len(rawData1)
    # rawData=[]
    # tester=[]
    # for i in range(0, noSeg):
    #     rawData.append(conversion(rawData1[i])[0])
    #     tester.append(conversion(tester1[i])[0])
    raw = getTargetVar(raw)
    rawData,var=conversion(raw)
    manager,rawData=getManager(rawData)
    tester,var=conversion(test)
    manager,tester = getManager(tester,manager)
    # for element in range(0,1):
    var=var+['high_manager','low_manager']
    pred=nnOutput(rawData,tester,1,var)

    # pool.close()
    # pool.join()
    # failures=[]
    # pred =None
    # for element in range(0,noSeg):
    #     got=que.get()
    #     if isinstance(got, int ):failures.append(got)
    #     elif pred is None :pred=got
    #     else:pred=pred.append(got)
    #
    # temp=rawData[failures[0]]
    # tempt=tester[failures[0]]
    # temp['location']=failures[0]
    # tempt['location']=failures[0]
    # fail=temp
    # son=tempt
    #
    #
    #
    # for i in range(1,len(failures)):
    #     temp = rawData[failures[i]]
    #     tempt = tester[failures[i]]
    #     temp['location'] = failures[i]
    #     tempt['location'] = failures[i]
    #     fail=fail.append(temp)
    #     son=son.append(tempt)
    # improvers=nnOutput(fail, son, -1,var)





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

