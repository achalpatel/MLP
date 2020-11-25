#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from enum import Enum

class Node:
    def __init__(self):
        self.inEdgeList = []
        self.outEdgeList = []
        self.value = None
    
    def addInEdge(self, edge):
        self.inEdgeList.append(edge)
    
    def addOutEdge(self, edge):
        self.outEdgeList.append(edge)

    def printData(self):
        print("value : ",self.value)
        for edge in self.inEdgeList:
            print("In - edge : ",edge)
        for edge in self.outEdgeList:
            print("Out - edge : ",edge)
        

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
    
    def printData(self):
        print("From Node : ", self.fromNode)
        print("To Node : ", self.toNode)
        print("Weight : ", self.weight)

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


class Graph:
    def __init__(self):
        self.inputLayer = InputLayer()
        self.outputLayer = OutputLayer()
        self.hiddenLayerList = []
        self.nodeList = []
    
    def createHiddenLayer(self):
        layer = HiddenLayer()        
        self.hiddenLayerList.append(layer)    
    
    def createInputNode(self):
        node = InputNode()
        self.inputLayer.addNode(node)
        self.nodeList.append(node)
        return node
    
    def createOutputNode(self):
        node = OutputNode()
        self.outputLayer.addNode(node)
        self.nodeList.append(node)
        return node
    
    def createHiddenNode(self, hiddenLayer: HiddenLayer):
        node = HiddenNode()
        hiddenLayer.addNode(node)
        self.nodeList.append(node)
        return node

    def createMultipleInputNodes(self, count : int):        
        for i in range(count):
            self.createInputNode()            

    def createMultipleOutputNodes(self, count : int):        
        for i in range(count):
            self.createOutputNode()            

    def createMultipleHiddenNodes(self, hiddenLayer : HiddenLayer, count : int):        
        for i in range(count):
            self.createHiddenNode(hiddenLayer)

    def connectInputToHidden(self):
        for fromNode in self.inputLayer.nodes:
            for toNode in self.hiddenLayerList[0].nodes:
                edge = Edge(fromNode, toNode)
                fromNode.addOutEdge(edge)
                toNode.addInEdge(edge)


    def connectHiddenToOutput(self):
        for fromNode in self.hiddenLayerList[0].nodes:
            for toNode in self.outputLayer.nodes:
                edge = Edge(fromNode, toNode)
                fromNode.addOutEdge(edge)
                toNode.addInEdge(edge)
    