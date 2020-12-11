#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random
import math

from pandas.core.frame import DataFrame


class Node:
    def __init__(self):
        self.inEdgeList = []
        self.outEdgeList = []
        self.value = None
        self.delta = None
    
    def addInEdge(self, edge):
        self.inEdgeList.append(edge)
    
    def addOutEdge(self, edge):
        self.outEdgeList.append(edge)

        
class InputNode(Node):
    pass

class OutputNode(Node):
    pass

class HiddenNode(Node):
    pass

class Edge:
    def __init__(self, fromNode, toNode):
        self.fromNode = fromNode
        self.toNode = toNode
        self.weight = None


class Layer:
    def __init__(self):
        self.nodes = []
    
    def addNode(self, node):
        self.nodes.append(node)
    def removeNode(self, node):
        self.nodes.remove(node)

class InputLayer(Layer):
    pass

class OutputLayer(Layer):
    pass

class HiddenLayer(Layer):
    pass

class Utility:
    def logistic(node: Node) -> float:
        sumValue = 0.0
        for edge in node.inEdgeList:            
            sumValue += edge.weight * edge.fromNode.value
        ans = 1.0 / (1.0 + math.exp(-sumValue))
        return ans

        

class Graph:
    def __init__(self, learningRate):
        self.inputLayer = InputLayer()
        self.outputLayer = OutputLayer()
        self.hiddenLayerList = []
        self.edgeList = []
        self.df = None
        self.trainDf = None
        self.testDf = None
        self.validationDf = None
        self.maxAttribList = []
        self.minAttribList = []
        self.targetList = []
        self.learningRate = learningRate
    
    def createHiddenLayers(self, count : int):
        for i in range(count):
            layer = HiddenLayer()        
            self.hiddenLayerList.append(layer)    
    
    def createNodes(self, inputNodeCount : int, hiddenNodeCount : int, outputNodeCount : int):
        self.createMultipleInputNodes(inputNodeCount)
        self.createMultipleHiddenNodes(hiddenNodeCount)
        self.createMultipleOutputNodes(outputNodeCount)    

    def createMultipleInputNodes(self, count : int):        
        for i in range(count):
            node = InputNode()         
            self.inputLayer.addNode(node)

    def createMultipleOutputNodes(self, count : int):        
        for i in range(count):
            node = OutputNode()
            self.outputLayer.addNode(node)            

    def createMultipleHiddenNodes(self, count : int):        
        for hiddenLayer in self.hiddenLayerList:            
            for i in range(count):
                node = HiddenNode()
                hiddenLayer.addNode(node)

    def connectLayers(self, fromLayer, toLayer):
        for fromNode in fromLayer.nodes:
            for toNode in toLayer.nodes:
                edge = Edge(fromNode, toNode)
                self.edgeList.append(edge)
                fromNode.addOutEdge(edge)
                toNode.addInEdge(edge)
    
    def connectInputToHidden(self):
        self.connectLayers(self.inputLayer, self.hiddenLayerList[0])
   
    def connectHiddenToHidden(self):
        for i in range(len(self.hiddenLayerList)-1):
            self.connectLayers(self.hiddenLayerList[i], self.hiddenLayerList[i+1])

    def connectHiddenToOutput(self):
        self.connectLayers(self.hiddenLayerList[-1], self.outputLayer)
    
    def connectGraph(self):
        self.connectInputToHidden()
        self.connectHiddenToHidden()
        self.connectHiddenToOutput()
        
    def calculateInitialWeights(self):
        n = len(self.hiddenLayerList[0].nodes)
        for edge in self.edgeList:
            edge.weight = random.uniform(-0.1, 0.1)

    def readDf(self, dataframe : DataFrame):
        self.df = dataframe.copy()        
        self.trainDf = self.df.sample(frac=0.8)
        self.testDf = self.df.drop(self.trainDf.index)    
        self.trainDf.reset_index(drop=True, inplace=True)
        self.testDf.reset_index(drop=True, inplace=True)     
        # self.balanceData()
        # Read Max, Min value of Each column and store in a List                
        for i in range(self.trainDf.shape[1]-1):
            self.maxAttribList.append(pd.DataFrame.max(self.trainDf.iloc[:,[i]]))
            self.minAttribList.append(pd.DataFrame.min(self.trainDf.iloc[:,[i]]))

    def balanceData(self):
        values, count = np.unique(self.trainDf['label'], return_counts=True)
        countList = list(count)
        valuesList = list(values)
        countDistribution = [(x/sum(countList))*100 for x in countList]
        print("Initial Data : ")
        print("countDistribution : ",countDistribution)
        print("countList : ",countList)

        countDataToAddList = [abs(max(countList)-x) for x in countList]
        print("countDataToAddList : ",countDataToAddList)
        for i in range(len(valuesList)):
            numberOfCopies = countDataToAddList[i]
            label_value = valuesList[i]
            row = self.trainDf.loc[self.trainDf['label'] == label_value]
            # print(row)
            if numberOfCopies>0:
                firstCopies = int(numberOfCopies/row.shape[0])
                remainderCopies = int(numberOfCopies % row.shape[0])
                newDf = pd.DataFrame()
                if firstCopies != 0:
                    newDf = pd.concat([row] * firstCopies)
                # if remainderCopies != 0:
                #     tempDf = row.iloc[ 0 : remainderCopies , : ]
                #     # print("tempDf:",tempDf)
                #     newDf = newDf.append(tempDf)
                self.trainDf = self.trainDf.append(newDf)
                # print("copies:",numberOfCopies,"row:",row.shape[0],"newDf:",newDf.shape[0],",df:",self.trainDf.shape[0])
                 

    
    def inputLayerFeed(self, row : DataFrame):        
        x = 0
        for node in self.inputLayer.nodes:
            normalized = float((row[x]-self.minAttribList[x])/(self.maxAttribList[x] - self.minAttribList[x]))
            node.value = normalized
            x+=1
        self.targetList = [0.2] * len(self.outputLayer.nodes)
        self.targetList[row[-1]] = 0.8        
    
    def softMax(self):
        denominator = self.softMaxDenominator()
        for node in self.outputLayer.nodes:
            node.value = math.exp(node.value)/denominator
            
    def softMaxDenominator(self) -> float:
        denominator = 0.0
        for node in self.outputLayer.nodes:
            denominator += math.exp(node.value)
        return denominator

    def mse(self):
        sumVal = 0.0
        m = len(self.outputLayer.nodes)
        for i in range(m):        
            sumVal += (self.targetList[i] - self.outputLayer.nodes[i].value) ** 2
        ans = sumVal/m
#         print("mse:",ans)
        return ans

    def deltaForOutputLayer(self):
        m = len(self.outputLayer.nodes)
        for i in range(m):
            node = self.outputLayer.nodes[i]
            deltaVal = node.value * (1-node.value) * (self.targetList[i]-node.value)
            node.delta = deltaVal
            
    def deltaForHiddenLayer(self):
        m = len(self.hiddenLayerList[0].nodes)
        for i in range(m):
            node = self.hiddenLayerList[0].nodes[i]
            sumVal = 0.0
            for edge in node.outEdgeList:
                sumVal += edge.toNode.delta * edge.weight
            deltaVal = node.value * (1-node.value) * sumVal
            node.delta = deltaVal

    def updateHiddenToOutputWeights(self):
        m = len(self.hiddenLayerList[0].nodes)
        for i in range(m):
            node = self.hiddenLayerList[0].nodes[i]
            for edge in node.outEdgeList:
                edge.weight += self.learningRate * edge.toNode.delta * node.value
    
    def updateInputToHiddenWeights(self):
        m = len(self.inputLayer.nodes)
        for i in range(m):
            node = self.inputLayer.nodes[i]
            for edge in node.outEdgeList:
                edge.weight += self.learningRate * edge.toNode.delta * node.value
    
    def singlePass(self, rowNumber) -> float:
        self.inputLayerFeed(self.trainDf.iloc[rowNumber])
        for layer in self.hiddenLayerList:
            for node in layer.nodes:
                node.value = Utility.logistic(node)
        
        for node in self.outputLayer.nodes:
            node.value = Utility.logistic(node)
        
        self.softMax()
        mseVal = self.mse()
        self.deltaForOutputLayer()
        self.deltaForHiddenLayer()
        self.updateInputToHiddenWeights()
        self.updateHiddenToOutputWeights()
        
        return mseVal
        
    def dfTest(self, dataframe):
        rightAnswerCount = 0
        outputIndexList = [0] * 8
        targetIndexList = [0] * 8
        for i in range(dataframe.shape[0]):
            self.inputLayerFeed(dataframe.iloc[i])
            for layer in self.hiddenLayerList:
                for node in layer.nodes:
                    node.value = Utility.logistic(node)

            for node in self.outputLayer.nodes:
                node.value = Utility.logistic(node)
            
            self.softMax()
            outputList = [node.value for node in self.outputLayer.nodes]
            outputIndex = outputList.index(max(outputList))
            targetIndex = self.targetList.index(max(self.targetList))

            # print("------------------------------------------------------")
            # print("outputList",outputList)
            # print("self.targetList",self.targetList)
            # print("Target Index : ", targetIndex)
            # print("Output Index:", outputIndex)
            outputIndexList[outputIndex] += 1
            targetIndexList[targetIndex] += 1
            if outputIndex == targetIndex:
                rightAnswerCount+=1
        print("targetIndexList : ",targetIndexList)
        print("outputIndexList : ",outputIndexList)
        print("rightAnswerCount : ",rightAnswerCount, "/Out of : ",dataframe.shape[0])

    def adaptiveLearning(self, t : int, n0 : float):
        alpha = 0.01
        self.learningRate = n0 * (math.exp(-alpha * t))
    
    def runANN(self, epochs : int):
        n0 = self.learningRate
        for k in range(epochs):
            mseSum = 0.0
            for i in range(self.trainDf.shape[0]):
                mseSum += self.singlePass(i)
            print("Epoch[",k,"] MSE:",mseSum,", LRate : ",self.learningRate)
            self.adaptiveLearning(k+1, n0)

    def printEdgeWeights(self):
        for edge in self.edgeList:
            print(edge.weight)