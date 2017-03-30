import numpy as np
import pandas as pd
from treatments import conversion
def sigmoid(x):
    z=x
    # z[z>15]=15
    # z[z<-15]=-15
    return np.exp(z)/(1+np.exp(z))


def sigDeriv(x):return sigmoid(x)*(1-sigmoid(x))
class layer(object):
    def __init__(self,cofficient):
        self.cofficient = cofficient
    def output(self,input):
        self.values=np.dot(input,self.cofficient)
        return self.values
    def gradientCorrection(self):pass
class neuralNetworks(object):#please add a base term in your data while passing the data
    def __init__(self,listOfMatrix,input,output,func,funcGradient,iteration=100):
        self.numLayers=len(listOfMatrix)
        self.layers=[]
        self.input=input
        self.func=func
        self.funcGradient=funcGradient
        self.actualOutput=output
        self.layerOutput=[]
        self.iteration=iteration
        for i in range(0,self.numLayers):
            self.layers.append(layer(listOfMatrix[i]))
    def feedForward(self,inputDataset=None):
        if inputDataset==None:input=self.input
        else :input=inputDataset
        self.layerOutput=[]
        for i in range(0, self.numLayers):
            input=self.func(self.layers[i].output(input))
            self.layerOutput.append(input)

        self.modelOutput=input
    def backwardPropagation(self,learningRate):
        self.change=[]
        self.cost=(-1.0/(self.modelOutput.shape[0]))*np.sum(self.actualOutput*(np.log((self.modelOutput)/np.sum(self.modelOutput,axis=1,keepdims=True))))
        self.change.append(learningRate*(-1.0/(self.modelOutput.shape[0]))*(np.dot(np.transpose(self.input),np.dot(((self.actualOutput/self.modelOutput)*self.funcGradient(self.layerOutput[1]))-\
                    ((1/np.sum(self.modelOutput,axis=1,keepdims=True))*self.funcGradient(self.layerOutput[1])),np.transpose(self.layers[1].cofficient))*self.funcGradient(self.layerOutput[0]))))
        self.change.append(learningRate * (-1.0 / (self.modelOutput.shape[0])) * (np.dot(np.transpose(self.layerOutput[0]),((self.actualOutput / self.modelOutput) * self.funcGradient(self.layerOutput[1])) - (
               (1 / np.sum(self.modelOutput, axis=1,keepdims=True)) * self.funcGradient(self.layerOutput[1])))))

        for i in range(0,len(self.change)):
            self.layers[i].cofficient-=self.change[i]
    def findEstimates(self):
        best=100
        for i in range(1,self.iteration):
            self.feedForward()

            if i<300:learningRate=2.0
            elif i<1000:learningRate=0.5
            elif i<4000:learningRate=0.2
            else :learningRate=0.1
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
        self.feedForward(X.as_matrix())
        predictedNormalize=self.modelOutput/np.sum(self.modelOutput,axis=1,keepdims=True)
        t = pd.DataFrame(predictedNormalize, columns=['high','medium','low'], index=summit.index)
        t['key'] = range(1, len(t) + 1)
        final = summit.merge(t, on='key', how='left')
        final = final[['listing_id', 'high', 'medium', 'low']]
        final = final.set_index('listing_id')
        final.to_csv(output, sep=',')





















