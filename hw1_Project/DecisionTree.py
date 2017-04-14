'''
Decision Tree based on ID3
INF552
@Yiming Liu
'''

from math import log
import operator
import sys
import string

#process dt-data.txt file and generate data list and label list
def createDataSet(filepath):
    f = open(filepath,'r+')
    line = f.readline()
    labels = dealString(line)
    dataSet = []
    while line:
        line = f.readline()
        if not line:
            break
        if(line !='\n'):
            dataSet.append(dealString(line))
    f.close()
    return dataSet, labels
#split strings and remove "\t" "\n" ";"
def dealString(strings):
    limitation = list(string.ascii_letters + ','+'-')
    tempItems = ""
    for item in strings:
        if len(item) == len([i for i in item if i in limitation]):
            tempItems += item
    return tempItems.split(',')

# use one attribute to split the dataset
def divideSet(dataSet, attrPos, attrVal):
    newDataSet  = []
    for attrVector in dataSet:
        if attrVector[attrPos] == attrVal:
            cutAttrVector = attrVector[:attrPos] #delete this attribute and save the new one into newDataSet
            cutAttrVector.extend(attrVector[attrPos+1:])
            newDataSet.append(cutAttrVector)
    return newDataSet

#calculate entropy for current dataset
def calEntropy(dataSet):
    itemNum = len(dataSet)
    labelCounts = {}

    for attrVector in dataSet:
        currentLabel = attrVector[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1


    Entropy = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key])/itemNum
        Entropy -= prob * log(prob,2)


    return Entropy

#calculate every attributes' IG and find the best attribute to split the dataset
#IG for attr i = original entropy - new entropy(remove attribute i from dataset)
def BestAttr(dataSet):
    AttrNum = len(dataSet[0])-1

    originEntropy = calEntropy(dataSet)
    bestIG = 0.0; bestAttr = -1
    for i in range(0,AttrNum):
        attrList = [buf[i] for buf in dataSet]


        Vals = set(attrList)
        newEntropy = 0.0
        for value in Vals:
            subDataSet = divideSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calEntropy(subDataSet)
        IG = originEntropy - newEntropy
        if (IG > bestIG):
            bestIG = IG
            bestAttr = i

    return bestAttr

def majorityCnt(classList):
    classCount = {}
    for buf in classList:
        if buf not in classCount.keys(): classCount[buf] = 0
        classCount[buf] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                                 key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def generateTree(dataSet,Labels):
    classList = [buf[-1] for buf in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:

        return majorityCnt(classList)

    bestAttr = BestAttr(dataSet)
    bestAttrLabel = Labels[bestAttr]
    myTree = {bestAttrLabel:{}}
    del(Labels[bestAttr])
    AttrValues = [exp[bestAttr] for exp in dataSet]
    uniqueVals = set(AttrValues)
    for value in uniqueVals:
        subAttrs = Labels[:]
        myTree[bestAttrLabel][value] = generateTree(divideSet\
        (dataSet, bestAttr, value),subAttrs)
    return myTree

if __name__ == '__main__':

    myDat, labels = createDataSet("dt-data.txt")
    myTree = generateTree(myDat, labels)
    print myTree

    #print calEntropy(myDat)
    #print BestAttr(myDat)

