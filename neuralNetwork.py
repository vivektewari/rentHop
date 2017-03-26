import numpy as np
def sigmoid(x):return 1/(1+np.exp(x)**-1)
def sigDeriv(x):return sigmoid(x)*(1-sigmoid(x))
class layer(object):
    def __init__(self,cofficient):
        self.cofficient = cofficient
    def output(self,input):
        return np.dot(self.cofficient,input)
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
            input=self.func(self.layers[i].output(input))
            self.layerOutput.append(self.layers[i].output(input))
        self.modelOutput=input
    def backwardPropagation(self):
        self.cost=self.actualOutput*(np.log(self.modelOutput)/np.dot(self.modelOutput,np.ones(shape=(3,1))))
        cofChange=np.dot(np.transpose(self.input)),(self.actualOutput/self.modelOutput)*self.funcGradient(self.layerOutput[-1])
        self.listOfMatrix[0]=self.listOfMatrix[0]-cofChange
    def findEstimates(self):
        for i in range(1,20):
            self.feedForward()
            self.backwardPropagation()
            print self.cost





















def neural(data,layers,learninRate):

def feedForward():
