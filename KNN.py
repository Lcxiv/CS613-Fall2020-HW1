# Example of kNN implemented in Python
import csv
import random
import math
import operator

def handleDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1,len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def minkowski_distance(a, b, p):
    return sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
            neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    result=[]
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1]== predictions[x]:
            correct += 1
    result.append((correct/float(len(testSet))) * 100.0 )
    result.append(correct)
    result.append(len(testSet))
    return result

def main(filename, split, k , bool=False):
    """This program takes three arguments and evaluates the data for K-NN
    filename -string- name of file to open
    split - float- the precet to split the data by
    k - int - k
    bool- boolean value- to decide whatever or not to persent the data
    """
    trainingSet=[]
    testSet=[]
    #split = 0.67
    handleDataset(filename, split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)) )
    print( 'Test set: ' + repr(len(testSet)))

    # generate predictions
    predictions=[]
    k = k
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        if bool==True:
            print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

    accuracy = getAccuracy(testSet, predictions)
    print()
    print('We have ' + repr(accuracy[1]) + ' correct, out of ' + repr(accuracy[2]) +'. Hence accuracy of: ' + repr(accuracy[0]) + '%')

#uncoment below to run the function
main('cars_v2.data.txt', 0.67 ,3, True)
