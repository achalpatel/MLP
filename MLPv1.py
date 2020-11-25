#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from enum import Enum


# In[2]:


class Node:
    def __init__(self):
        self.inEdgeList = []
        self.outEdgeList = []
    
    def addInEdge(edge):
        self.inEdgeList.append(edge)
    
    def addOutEdge(edge):
        self.outEdgeList.append(edge)
    


# In[3]:


class InputNode(Node):
    pass


# In[4]:


class OutputNode(Node):
    def __init__(self):
        self.value = None


# In[5]:


class HiddenNode(Node):
    def __init__(self):
        self.value = None


# In[6]:


class Edge:
    def __init__(self, fromNode, toNode):
        self.fromNode = fromNode
        self.toNode = toNode
        self.weight = None


# In[7]:


class Layer:
    def __init__(self):
        self.nodes = []
    
    def addNode(node):
        self.nodes.append(node)
    def removeNode(node):
        self.nodes.remove(node)


# In[8]:


class InputLayer(Layer):
    pass


# In[9]:


class OutputLayer(Layer):
    pass


# In[10]:


class HiddenLayer(Layer):
    pass


# In[11]:


class NodeEnum(Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    HIDDEN = "HIDDEN"


# In[16]:


class Graph:
    def __init__(self):
        self.inputLayer = None
        self.outputLayer = None
        self.hiddenLayerList = []
        self.nodeList = []
    
    def createInputLayer(self):
        layer = InputLayer()
        self.inputLayer = layer
    
    def createOutputLayer(self):
        layer = OutputLayer()
        self.outputLayer = layer
    
    def createHiddenLayer(self):
        layer = HiddenLayer()        
        self.hiddenLayerList.append(layer)
    
    def createInputNode(self):
        node = InputNode()
        self.nodeList.append(node)
        return node
    
    def createOutputNode(self):
        node = OutputNode()
        self.nodeList.append(node)
        return node
    
    def createHiddenNode(self):
        node = HiddenNode()
        self.nodeList.append(node)
        return node
    
    


# In[17]:


g = Graph()
g.createInputLayer()
g.createOutputLayer()
g.createHiddenLayer()
g.createInputNode()
g.createOutputNode()
g.createHiddenNode()


# In[ ]:




