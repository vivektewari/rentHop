from multiprocessing import Pool,Process,cpu_count,Manager
import time
import copy
import json,csv
import pandas as pd
from neuralNetwork import  neuralNetworks as nn
import numpy as np
import equation
from segmentation import makeCart
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
    secondLayer=5
    d=nn(listOfMatrix=[np.random.rand(len(var),secondLayer),np.random.rand(secondLayer,3)],input=rawTransformed[var].as_matrix(),output=rawTransformed[['high','medium','low']].as_matrix(),func=sigmoid,funcGradient=sigDeriv,variables=var,iteration=500)
    d.findEstimates()
    print d.cost,d.input.shape[0]
    if analyse:
        analyseData1 = analyseData.copy(deep=True)
        temp = d.analyseObservation(dataSet=analyseData1)
        return temp
    prediction=d.predict(test=test)

    return prediction,d.cost,d.input.shape[0]


raw=pd.read_json('input\\train.json')

test=pd.read_json('input\\test.json')
r=copy.deepcopy(raw)
t=copy.deepcopy(test)

print "dataset brought to python"

raw = getTargetVar(raw)
rawData,var=conversion(raw)
manager,rawData=getManager(rawData)
tester,var=conversion(test)
manager,tester = getManager(tester,manager)
# for element in range(0,1):
var=var  +['high_manager','low_manager']

#split 1 price >=3600  17632, further split in price didnt ring better result,2.low_manager>0.7303
splitter=1300
# rawData.update(r)
rawData=rawData[r['price']<3100]
r=r[r['price']<3100]
# rindexes=rawData.query('high_manager>0').index
# tindexes=tester.query('high_manager>0').index
# pred = nnOutput(rawData.loc[rindexes,:], tester.loc[tindexes,:], 1, var,True)
# # pred=nnOutput(rawData, tester, 1, var,True)
# pred.update(r)
# pred=pred[var+['high','medium','low','cost']]
# pred.to_csv('output\\highnalysis.csv', sep=',')
# for i in range(20,40):
#     split=splitter+i*50
#     print split
#     pred,cost1,row1=nnOutput(rawData[r['price']<split],tester[t['price']<split],1,var)
#     pred, cost2, row2=nnOutput(rawData[r['price']>=split],tester[t['price']>=split],1,var)
#     print (cost1*row1+cost2*row2)/31720
#from boost import nnOutput
var=var +['high_manager','low_manager']
r=r[rawData['low_manager']<=0.7303]
rawData=rawData[rawData['low_manager']<=0.7303]

tester=rawData
rindexes=rawData.query('low_manager <=0.5084').index
tindexes=tester.query('low_manager <= 0.5084').index
# rawData1=rawData.loc[rindexes,:]
# rawData2=rawData.drop(rindexes)

pred1= nnOutput(rawData.loc[rindexes,:], tester.loc[tindexes,:], 1, var,True)
pred2= nnOutput(rawData.drop(rindexes), tester.drop(tindexes), 1, var,True)
pred=pred1.append(pred2)
pred.update(r)
pred=pred[var+['high','medium','low','cost']]
pred.to_csv('output\\highnalysis.csv', sep=',')
#print (cost1*row1+cost2*row2)/13826
# p=makeCart(rawData.loc[rindexes,var].as_matrix(),rawData.loc[rindexes,['high','medium','low']].as_matrix(),var)

#
#
#
#
#
# pred=pred.sort_index()
# print pred.shape[0]
# print np.sum(pred.as_matrix())
# pred.to_csv('output\\test.csv', sep=',')
# timeTaken = time.time() - start_time
# print "time taken is"
# print timeTaken



