import numpy as np
from mathFunctions import sigmoid,sigDeriv
from treatments import getTargetVar
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
    def nniterate(self):
        classi=self.classifier
        classi.weight = (1 / float(classi.actualOutput.shape[0])) * np.ones(shape=(classi.actualOutput.shape[0], 1))
        for i in range(0,self.maxIteration):
            classi.findEstimates()
            classifierWeight=np.log((2.5-classi.weightedCost)/classi.weightedCost)
            tes=(classi.predict(self.test)).as_matrix()
            self.prediction=tes*classifierWeight+self.prediction
            self.classifierWeight.append(classifierWeight)

            costMatrix=(self.classifier.analyseObservation(self.trainCopy))['cost'].as_matrix()
            y=sigmoid(classifierWeight*(costMatrix.reshape((costMatrix.shape[0],1))>0.55).astype(int))
            newWeight=classi.weight*y
            classi.weight=newWeight/sum(newWeight)

        return self.prediction














