import numpy as np
import pandas as pd
import copy
from treatments import conversion,getTargetVar

from math import log


class layer(object):
    def __init__(self,cofficient):
        self.cofficient = cofficient
    def output(self,input):
        self.values=np.dot(input,self.cofficient)
        if (abs(self.values)>600).any():
            T=0
        return self.values
    def gradientCorrection(self):pass
class neuralNetworks(object):#please add a base term in your data while passing the data
    def __init__(self,listOfMatrix,input,output,func,funcGradient,variables,iteration=100,weight=None):
        self.variables=variables
        self.numLayers=len(listOfMatrix)
        self.layers=[]
        self.input=input
        self.func=func
        self.funcGradient=funcGradient
        self.actualOutput=output
        self.layerOutput=[]
        if weight is None:
            rows=output.shape[0]
            cols=output.shape[1]
            self.weight=(1/float(rows))*np.ones((rows,1))
        else :self.weight=weight
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
        self.weightedCost=-np.sum(self.actualOutput*self.weight*(np.log((self.modelOutput)/np.sum(self.modelOutput,axis=1,keepdims=True))))
        self.change.append(learningRate*-(np.dot(np.transpose(self.input),np.dot(((self.actualOutput*self.weight/self.modelOutput)*self.funcGradient(self.layerOutput[1]))-\
                    ((self.weight/np.sum(self.modelOutput,axis=1,keepdims=True))*self.funcGradient(self.layerOutput[1])),np.transpose(self.layers[1].cofficient))*self.funcGradient(self.layerOutput[0]))))
        self.change.append(learningRate * -(np.dot(np.transpose(self.layerOutput[0]),((self.actualOutput*self.weight / self.modelOutput) * self.funcGradient(self.layerOutput[1])) - (
               (self.weight / np.sum(self.modelOutput, axis=1,keepdims=True)) * self.funcGradient(self.layerOutput[1])))))

        for i in range(0, len(self.change)): self.layers[i].cofficient -= self.change[i]

    def findEstimates(self):
        best=100
        bestCof=[]
        for i in range(0, len(self.layers)):bestCof.append(copy.deepcopy(self.layers[i].cofficient))
        self.feedForward()
        for i in range(1,self.iteration):
            if i<100:learningRate=0.2
            else:learningRate=0.1
            self.backwardPropagation(learningRate)
            self.feedForward()
            if (np.isnan(self.modelOutput)).any():
                break
            elif self.cost<best:
                besttemp = []
                for j in range(0,len(self.layers)):besttemp.append(self.layers[j].cofficient)
                bestCof=copy.deepcopy(besttemp)
                best=self.cost
            else:
                continue
        self.cost=best
        for k in range(0, len(self.layers)):
            self.layers[k].cofficient=bestCof[k]

    def analyseObservation(self,dataSet):
        analyseData = getTargetVar(dataSet)
        final=self.predict(analyseData)

        analyseData['cost']=-np.sum(dataSet[['high','medium','low']].as_matrix()*np.log(final[['high','medium','low']].as_matrix()),axis=1)
        if analyseData.isnull().values.any():
            r=0
        return analyseData



    def predict(self,test,first=True):

        X = test[:].loc[:, self.variables]
        self.feedForward(X.as_matrix())
        predictedNormalize=self.modelOutput/np.sum(self.modelOutput,axis=1,keepdims=True)
        final = pd.DataFrame(predictedNormalize, columns=['high','medium','low'], index=test.listing_id)



        return final





















