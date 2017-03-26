import numpy as np
import pandas as pd
def sigmoid(x):return 1/(1+np.exp(x)**-1)
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
    def backwardPropagation(self):
        y=(self.modelOutput.shape[0])
        x=self.actualOutput*(np.log((self.modelOutput)/np.dot(self.modelOutput,np.ones(shape=(3,1)))))
        self.cost=(-1.0/(self.modelOutput.shape[0]))*np.sum(self.actualOutput*(np.log((self.modelOutput)/np.dot(self.modelOutput,np.ones(shape=(3,1))))))
        cofChange=0.3*(-1.0/(self.modelOutput.shape[0]))*np.dot(np.transpose(self.input),(self.actualOutput/self.modelOutput)*self.funcGradient(self.layerOutput[-1]))
        self.layers[0].cofficient-=cofChange
    def findEstimates(self):
        for i in range(1,200):
            self.feedForward()
            self.backwardPropagation()
            print self.cost
    def predict(self,test,output):
        summit = test[['listing_id']]
        summit['key'] = range(1, len(summit) + 1)
        t = pd.DataFrame(self.modelOutput, columns=['high','medium','low'], index=summit.index)
        t['key'] = range(1, len(t) + 1)
        final = summit.merge(t, on='key', how='left')
        final = final[['listing_id', 'high', 'medium', 'low']]
        final = final.set_index('listing_id')
        final.to_csv(output, sep=',')





















