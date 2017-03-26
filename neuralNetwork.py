import numpy as np
import pandas as pd
from treatments import conversion
def sigmoid(x):

    return np.exp(x)/(1+np.exp(x))

def sigDeriv(x):return sigmoid(x)*(1-sigmoid(x))
class layer(object):
    def __init__(self,cofficient):
        self.cofficient = cofficient
    def output(self,input):
        return np.dot(input,self.cofficient)
    def gradientCorrection(self):pass
class neuralNetworks(object):
    def __init__(self,listOfMatrix,input,output,func,funcGradient):
        self.numLayers=len(listOfMatrix)
        self.layers=[]
        self.input=input
        self.func=func
        self.funcGradient=funcGradient
        self.actualOutput=output
        self.layerOutput=[]
        for i in range(0,self.numLayers):
            self.layers.append(layer(listOfMatrix[i]))
    def feedForward(self):
        input=self.input
        for i in range(0, self.numLayers):
            self.layerOutput.append(self.layers[i].output(input))
            input=self.func(self.layers[i].output(input))

        self.modelOutput=input
    def backwardPropagation(self,learningRate):
        self.cost=(-1.0/(self.modelOutput.shape[0]))*np.sum(self.actualOutput*(np.log((self.modelOutput)/np.dot(self.modelOutput,np.ones(shape=(3,1))))))
        cofChange=learningRate*(-1.0/(self.modelOutput.shape[0]))*(np.dot(np.transpose(self.input),((self.actualOutput/self.modelOutput)*self.funcGradient(self.layerOutput[-1]))-((self.actualOutput/np.sum(self.modelOutput))*np.dot(self.funcGradient(self.layerOutput[-1]),np.ones(shape=(3,1))))))
        self.layers[0].cofficient-=cofChange
    def findEstimates(self):
        best=100
        for i in range(1,2000):
            self.feedForward()
            learningRate=10.0/i
            self.backwardPropagation(learningRate)
            if self.cost<best:
                best=self.cost
                bestCof=self.layers[0].cofficient
            print self.cost
        self.layers[0].cofficient=bestCof
        print best
    def predict(self,test,output='output\\test.csv'):
        summit = test[['listing_id']]
        df1, variables = conversion(test)
        X = df1[:].loc[:, variables]
        summit['key'] = range(1, len(summit) + 1)
        t=np.sum(self.func(self.layers[0].output(X.as_matrix())),axis=1)
        predicted=self.func(self.layers[0].output(X.as_matrix()))/np.dot(self.func(self.layers[0].output(X.as_matrix())),np.ones(shape=(3,1)))
        t = pd.DataFrame(predicted, columns=['high','medium','low'], index=summit.index)
        t['key'] = range(1, len(t) + 1)
        final = summit.merge(t, on='key', how='left')
        final = final[['listing_id', 'high', 'medium', 'low']]
        final = final.set_index('listing_id')
        final.to_csv(output, sep=',')





















