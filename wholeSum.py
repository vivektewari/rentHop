from multiprocessing import Pool,Process,cpu_count,Manager
from scipy.optimize import minimize
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
    secondLayer=8
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

test=pd.read_json('input\\test.json')[0:1000]
r=copy.deepcopy(raw)
t=copy.deepcopy(test)

print "dataset brought to python"

raw = getTargetVar(raw)
rawData,var=conversion(raw)
manager,rawData=getManager(rawData)
tester,var=conversion(test)
manager,tester = getManager(tester,manager)
# for element in range(0,1):
var=var +['high_manager','low_manager']

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
#
r=r[rawData['low_manager']<=0.7303]
rawData=rawData[rawData['low_manager']<=0.7303]
rawData['listing_id']=(rawData['listing_id']-rawData['listing_id'].mean())/(rawData['listing_id'].max()-rawData['listing_id'].min())
var=var+['listing_id']
tester=rawData

# rindexes=rawData.query('low_manager <=0.5084').index
# tindexes=tester.query('low_manager <= 0.5084').index
# # rawData1=rawData.loc[rindexes,:]
# # rawData2=rawData.drop(rindexes)
#
# pred1= nnOutput(rawData.loc[rindexes,:], tester.loc[tindexes,:], 1, var,True)
# pred2= nnOutput(rawData.drop(rindexes), tester.drop(tindexes), 1, var,True)
# pred=pred1.append(pred2)
# pred.update(r)
# pred=pred[var+['high','medium','low','cost']]
# pred.to_csv('output\\highnalysis.csv', sep=',')

# y=data**2
# data=np.hstack((data,y))
# si=rawData.shape[0]
# sq=[]
# for element in var:
#     sq.append(element+"_sq")
# var=var+sq
def costMatrix(p):
    mid=len(p)/2
    p1=p[0:mid]
    p2=p[mid:2*mid]
    prob=np.transpose(np.matrix(np.vstack((p1,p2))))
    a=(sigmoid(np.dot(data,prob)))
    b=1-np.sum(a,axis=1)
    probability=np.matrix(np.hstack((a,b)))
    logp=np.log(probability)
    cost=-np.sum(np.multiply(target, logp),axis=1)
    return cost
def fun(p):
    cost = costMatrix(p)
    return (np.sum(cost,axis=0))[0,0]/cost.shape[0]


p=np.array([-7.06453423 ,-0.44768188  ,0.19550523 ,-0.40279784 ,-0.02602051  ,0.25124472,
  0.38361667, -1.64564863 ,-0.30294213 , 0.07838408 , 0.02767179 , 1.28916747,
  0.46313221,  1.05722549 , 0.22053143,  0.23369542 , 0.20174279 , 1.51846885,
  0.18636913, -0.64736853, -0.05645325 , 0.05044996 , 0.87177072,  7.92897264,
 -3.82256583,  0.6743756  , 3.88470644 , 0.94427169 ,-0.05686988 ,-0.29754628,
  0.10287585, -0.29683828 ,-0.03877167 , 2.15075129 , 0.52831058 ,-2.96652503,
 -0.22941295 , 0.92337195 , 1.72542233 , 1.09565885 , 2.10501794 , 1.49306637,
  1.59396277  ,0.66082587,  1.54074395 , 1.32979109 , 1.86296696 , 1.95980294,
  1.33316042 ,-8.26323114 ,-9.07219985 ,-1.50869016])
bad=pd.read_csv('output\\bad2.csv')
bad['match']=bad['listing_id']
bad=bad.set_index(['match'])
rawData['match']=rawData['listing_id']
rawData=rawData.set_index(['match'])
bad.index=bad.index.map(str)
rawData.index=rawData.index.map(str)
#rawData[var].to_csv('output\\13826.csv')

tar=rawData.drop(bad.index.values)

data = tar[var].as_matrix()
target=tar[['high', 'medium', 'low']].as_matrix()
print (fun(p))
# rn=np.log(np.transpose(np.matrix([rawData['high'].mean(),rawData['medium'].mean(),rawData['low'].mean()])))
# print -np.sum(np.dot(rawData[['high','medium','low']].as_matrix(),rn))/rawData.shape[0]
#f=fun(np.random.rand(len(var)*2))
def sum1(p):
    mid=len(p)/2
    p1=p[0:mid]
    p2=p[mid:2*mid]
    prob=np.transpose(np.matrix(np.vstack((p1,p2))))
    t=np.squeeze(np.array(np.ones(shape=(data.shape[0], 1)) - np.sum((sigmoid(np.dot(data, prob))), axis=1)))
    #print t
    return t
cons = ({'type': 'ineq', 'fun': lambda p: sum1(p)},
        {'type': 'ineq',
         'fun': lambda p: np.squeeze(np.array(sigmoid(np.dot(data, p[0:len(var)].reshape(len(var), 1)))))},
        {'type': 'ineq', 'fun': lambda p: np.squeeze(
            np.array(sigmoid(np.dot(data, p[len(var):2 * len(var)].reshape(len(var), 1)))))})



# t= np.random.rand(len(var),3)
# r=np.sum(sigmoid(np.dot(data,t)),axis=1,keepdims=True)-np.ones(shape=(data.shape[0],1))
pinitialize=np.random.rand(len(var)*2)
pinitialize[0]=-6.9
pinitialize[0+len(var)]=-6.9
data = rawData[var].as_matrix()
target=rawData[['high', 'medium', 'low']].as_matrix()
excluded=None
for i in range(1,80):

    res = minimize(fun, pinitialize, method='SLSQP',  constraints=cons)
    r = costMatrix(res.x)
    if not np.isnan(r).any():
        temp = np.hstack((r,data,target))
        temp=temp[temp[:,0].argsort(axis=0)]
        temp=np.squeeze(temp,axis=(1))
        data=temp[0:int(temp.shape[0]-0.01*temp.shape[0]),1:temp.shape[1]-3]
        target=temp[0:int(temp.shape[0]-0.01*temp.shape[0]),temp.shape[1]-3:temp.shape[1]]
        if excluded==None:excluded = temp[int(temp.shape[0] - 0.01 * temp.shape[0]):, 0:temp.shape[1] ]
        else  :excluded=np.vstack((excluded,temp[int(temp.shape[0] - 0.01 * temp.shape[0]):, 0:temp.shape[1] ]))
        final=res.x
        pinitialize=res.x


    else:
        pinitialize = np.random.rand(len(var) * 2)
        pinitialize[0] = -6.9
        pinitialize[0 + len(var)] = -6.9

    print data.shape[0],res.fun,i
bad=pd.DataFrame(excluded,columns=['cost']+var+['high','medium','low'])
bad.to_csv('output\\bad.csv', sep=',')
timeTaken = time.time() - start_time
print final
print (fun(final))
print "time taken is"
print timeTaken
#print (sigmoid(np.dot(data,np.matrix((res.x).reshape(len(var),2)))))
#
# r=costMatrix(res.x)
# e=sorter(r)

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

