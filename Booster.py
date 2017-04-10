import numpy as np
from mathFunctions import sigmoid,sigDeriv
from treatments import getTargetVar
import pandas as pd
class booster(object):
    """
    assignes weights to each observation
    calculates weighted loss
    saves classifier prediction and weight of the classifier
    finally returns weighted average of classifier
    """
    def __init__(self,maxIteration,classifier,test,trainCopy):#func api is that it returns prediction actialCost and weightedCost
        self.classifier=classifier
        self.maxIteration=maxIteration
        self.prediction=np.zeros(shape=(test.shape[0],3))
        self.error=[]
        self.weightedError=[]
        self.test=test
        self.classifierWeight=[]
        self.trainCopy=trainCopy
    def nniterate(self,train=False):
        classi=self.classifier
        classi.weight = (1 / float(classi.actualOutput.shape[0])) * np.ones(shape=(classi.actualOutput.shape[0], 1))
        for i in range(0,self.maxIteration):
            for j in range(0,len(classi.layers)):classi.layers[j].cofficient=np.random.rand(classi.layers[j].cofficient.shape[0],classi.layers[j].cofficient.shape[1])
            classi.findEstimates()
            classifierWeight=np.log((2.5-classi.cost)/classi.cost)
            tes=classi.predict(self.test)
            self.prediction =tes*classifierWeight+self.prediction
            self.classifierWeight.append(classifierWeight)

            costMatrix=(self.classifier.analyseObservation(self.trainCopy))['cost'].as_matrix()
            y=sigmoid(classifierWeight*(costMatrix.reshape((costMatrix.shape[0],1))>0.55).astype(int))
            if (np.isnan(y)).any():print "weight comes out to be nan"
            newWeight=classi.weight*y
            classi.weight=newWeight/sum(newWeight)
        self.prediction = self.prediction.div(np.sum(self.prediction, axis=1), axis=0)
        pred=self.prediction
        if train:
            self.trainCopy.index=self.prediction.index
            pred=self.prediction.join(self.trainCopy,how='left',rsuffix='_t')
            pred['cost']=np.log(pred['high'])*pred['high_t']+np.log(pred['medium'])*pred['medium_t']+np.log(pred['low'])*pred['low_t']
            print np.sum(pred['cost'])/self.trainCopy.shape[0],self.trainCopy.shape[0]

        return pred
    def weightSelecter(self,train=False):
        classi=self.classifier
        classi.weight = (1 / float(classi.actualOutput.shape[0])) * np.ones(shape=(classi.actualOutput.shape[0], 1))
        best=5.0
        for i in range(0,self.maxIteration):
            for j in range(0,len(classi.layers)):classi.layers[j].cofficient=np.random.rand(classi.layers[j].cofficient.shape[0],classi.layers[j].cofficient.shape[1])
            classi.findEstimates()
            if classi.cost<best:
                best=classi.cost
                self.prediction = classi.predict(self.test)

            classifierWeight=np.log((2.5-classi.cost)/classi.cost)
            self.classifierWeight.append(classifierWeight)
            costMatrix=(self.classifier.analyseObservation(self.trainCopy))['cost'].as_matrix()
            y=sigmoid(classifierWeight*(costMatrix.reshape((costMatrix.shape[0],1))>0.55).astype(int))
            if (np.isnan(y)).any():print "weight comes out to be nan"
            newWeight=classi.weight*y
            classi.weight=newWeight/sum(newWeight)

        print best,classi.input.shape[0]
        return self.prediction















