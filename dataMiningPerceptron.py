import numpy as np
from pandas import read_csv

# import training data and test data and store as arrays
train = np.asarray(read_csv("train.data", header = None))
test = np.asarray(read_csv("test.data", header = None))

# add in first column of data (x_0) as 1 then can use first element of weights vector (w_0) as bias
biasTrain = np.insert(train, 0, 1, axis=1)
biasTest = np.insert(test, 0, 1, axis=1)


def removeExtraClass(inputData, extraClass:str):
    """
    Creates separate arrays for each pair of classes, ignoring the third.
    
    Parameters:
        inputData - (ndarray) array of data
        extraClass - (string) name of class to be disgarded
    Returns:
        editedData - (ndarray) array of data containing just 2 classes
    """
    editedData = np.asarray([x for x in inputData if extraClass not in x])
    return editedData


def positiveClass(inputData, posClass:str):
    """
    Take only the last column of classes and set the positive class to 1 
    All others are assigned -1.
    
    Parameters:
        inputData - (ndarray) array of data
        posClass - (string) name of class to be assigned 1
    Returns:
        labels - (ndarray) array of 1s and -1s set to variable name specified when calling the method
    """
    labels = inputData [:,[-1]]
    return np.where(labels == posClass, 1, -1)


def dataOnly(inputData):
    """
    Take the data array and remove the class labels.
    
    Parameters:
        inputData - (ndarray) array of data
    Returns:
        inputData - (ndarray) array of data without the class labels
    """
    return np.delete(inputData, -1, 1)


"""
The following method calls split the data into binary pairs, removing the third unwanted set which is specified.
"""
# class 1 and class 2
train_1_2 = removeExtraClass(biasTrain, "class-3")
test_1_2 = removeExtraClass(biasTest, "class-3")

# class 2 and class 3
train_2_3 = removeExtraClass(biasTrain, "class-1")
test_2_3 = removeExtraClass(biasTest, "class-1")

# class 1 and class 3
train_1_3 = removeExtraClass(biasTrain, "class-2")
test_1_3 = removeExtraClass(biasTest, "class-2")


"""
The following method calls set the positive class label as 1 and the negative class as -1 for the binary data in Question 3.
The data and labels are then split in order to be clearly accessed as separate arrays within the perceptron algorithms.
"""
# class 1 and class 2
trainLabel_1_2  = positiveClass(train_1_2, "class-1")
trainData_1_2 = dataOnly(train_1_2)
testLabel_1_2  = positiveClass(test_1_2, "class-1") 
testData_1_2 = dataOnly(test_1_2)

# class 2 and class 3
trainLabel_2_3  = positiveClass(train_2_3, "class-2") 
trainData_2_3 = dataOnly(train_2_3)
testLabel_2_3  = positiveClass(train_2_3, "class-2")
testData_2_3 = dataOnly(train_2_3)

# class 1 and class 3
trainLabel_1_3  = positiveClass(train_1_3, "class-1") 
trainData_1_3 = dataOnly(train_1_3)
testLabel_1_3  = positiveClass(test_1_3, "class-1")
testData_1_3 = dataOnly(test_1_3)


"""
The following data manipulation is for the one-vs-rest approach stated in Question 4.
The positive class is assigned labels of 1 whereas the other two classes are assigned -1.
The labels are also separately assigned 0, 1 and 2 in order to match with their indexes and be used in the multi-class classifier.
"""
multiTrainLabel_1 = positiveClass(biasTrain, "class-1")
multiTrainLabel_2 = positiveClass(biasTrain, "class-2")
multiTrainLabel_3 = positiveClass(biasTrain, "class-3")

multiTrainDataAll = dataOnly(biasTrain)
multiTrainLabelAll = biasTrain [:,[-1]]
multiTrainLabelAll = np.asarray([0 if x == "class-1" else 1 if x == "class-2" else 2 if x == "class-3" else x for x in multiTrainLabelAll])

multiTestDataAll = dataOnly(biasTest)
multiTestLabelAll = biasTest [:,[-1]]
multiTestLabelAll = np.asarray([0 if x == "class-1" else 1 if x == "class-2" else 2 if x == "class-3" else x for x in multiTestLabelAll])


def shuffle(array1, array2):
    """
    Shuffles 2 arrays in unison so data still correlates with the correct label
    Parameters:
        array1 - (ndarray) array of data
        array2 - (ndarray) array of labels
    Returns:
        array1[perm] - (ndarray) a permutation of the data array shuffled in a random order
        array2[perm] - (ndarray) a permutation of the labels array shuffled in the same order as the data array
    """
    perm = np.random.permutation(len(array1))
    return array1[perm], array2[perm]


def perceptronTrain(trainData, trainLabel, maxEpochs, regCoefficient):
    """
    Training perceptron algorithm.
    Uses training data and corresponding labeled classes to learn to classify binary data.
    Bias is included with the weights and is the first element within then weights array.
    Weights are updated whenever a misclassification is made in order to correct the error.
    
    Parameters:
        trainData - (ndarray) array of data
        trainLabel - (ndarray) array of labels corresponding to data
        maxEpochs - (int) the number of iterations to be undertaken
        regCoefficient - (float) L2 regularisation coefficient
    Returns:
        weights  - (ndarray) the final weights vector produced after training
        predList  - (list) a list of the predictions generated. Used for accuracy evaluation
        trueList - (list)  a list of the labels corresponding to the predictions. Used for accuracy evaluation
    """
    # initialise weights vector (set to zero) of same length as data set(including bias)
    weights = np.zeros(len(trainData[0]))

    # empty lists for predictions to populate
    predList = []
    trueList = []

    # set random number to zero at beginning of run
    s = 0
    np.random.seed(s)

    for epoch in range (maxEpochs):
        # increment the seed used to shuffle data and labels in unison
        s +=1
        trainData, trainLabel = shuffle(trainData, trainLabel)

        # calculate the activation score for each element of data
        for idx, x_i in enumerate(trainData):
            aScore = np.inner(weights, x_i)
            if aScore > 0:
                predict = 1
            else:
                predict = -1

            # add prediction and corresponding true label to lists for accuracy evaluation
            predList.append(predict)
            trueList.append(trainLabel[idx])

            # check if prediction matches actual value
            check = trainLabel[idx] * predict

            # if negative value (misclassification) then update weights
            if check <= 0:
                for idw, w_i in enumerate(weights):
                    # perceptron update rule including L2 regularisation
                    weights[idw] = (1 - 2 * regCoefficient) * weights[idw] + trainLabel[idx] * x_i[idw]

    return weights, predList, trueList


def perceptronTest(weights, testData, testLabel):
    """
    Testing perceptron algorithm.
    Uses the weights produced from the training algorithm to classify test data.
    
    Parameters:
        weights - (ndarray) weights vector produced from the training algorithm
        testData - (ndarray) array of data
        testLabel - (ndarray) array of labels corresponding to data
    Returns:
        predList  - (list) a list of the predictions generated. Used for accuracy evaluation
        trueList - (list) a list of the labels corresponding to the predictions. Used for accuracy evaluation
    """
    # empty lists for predictions to populate
    predList = []
    trueList = []

    # run the same perceptron algorithm but using the weights output by training
    for idx, x_i in enumerate(testData):
        aScore = np.inner(weights, x_i)
        if aScore > 0:
            predict = 1
        else:
            predict = -1

        # add prediction and corresponding true label to lists for accuracy evaluation
        predList.append(predict)
        trueList.append(testLabel[idx])
    return predList, trueList


def multiClassifier(modelA, modelB, modelC, multiData, multiLabel):
    """
    Multi-class perceptron algorithm.
    Uses the models produced from running one-vs-rest style data through the training algorithm.
    
    Parameters:
        multiData - (ndarray) array of data
        multiLabel - (ndarray) array of labels corresponding to data
    Returns:
        predList  - (list) a list of the predictions generated. Used for accuracy evaluation
        trueList - (list) a list of the labels corresponding to the predictions. Used for accuracy evaluation
    """
    # empty lists for predictions to populate
    predList = []
    trueList = []

    # input models produced from training as array
    modelsArray = np.array([modelA, modelB, modelC], dtype=float)

    for idx, x_i in enumerate(multiData):
        aScore = []
        for model in modelsArray:
            # generate activation score as measure of confidence in classification with each model and add to list
            aScore.append(np.inner(model, x_i))
            # choose the highest activation score from list as prediction
            predict = np.argmax(aScore)

        # add prediction and corresponding true label to lists for accuracy evaluation
        predList.append(predict)
        trueList.append(multiLabel[idx])
    return predList, trueList


def convert(outputList):
    """
    Convert lists outputted by perceptron to 1D arrays.
    Positive class remains as 1s but negative class is converted to 0s.
    This is so np.count_nonzero can be used for the evaluation.
    
    Parameters:
        outputList - (list) list of predictions or labels as 1s and -1s
    Returns:
        outputArray  - (ndarray) a 1D array of 1s and 0s
    """
    # convert list to 1D arrays
    outputArray = np.reshape(outputList,-1)
    # convert -1s to 0s
    outputArray[outputArray==-1] = 0
    return outputArray


def calcAccuracy(trainOrTest:str, predArray, trueArray):
    """
    Counts the true positives, true negatives, false positives and false negatives
    of the predictions made by the algorithm and then computes the accuracy.
    
    Parameters:
        trainOrTest - (string) name of data being evaluated. Printed with the accuracy.
        predArray - (ndarray) list of predictions outputted by one of the perceptron algorithms
        trueArray - (ndarray) list of labels corresponding to predictions
    Returns:
        print  - (function) the accuracy printed with the relevant name of the data
    """
    # convert lists to 1D arrays with 0s instead of -1s
    predArray = convert(predArray)
    trueArray = convert(trueArray)

    # create masks
    posMaskTest = trueArray == 1
    negMaskTest = trueArray == 0

    # count true positives, false negatives, false positives and true negatives
    truePos = np.count_nonzero(predArray[posMaskTest] ==1)
    falseNeg = np.count_nonzero(predArray[posMaskTest] ==0)
    falsePos = np.count_nonzero(predArray[negMaskTest] ==1)
    trueNeg = np.count_nonzero(predArray[negMaskTest] ==0)

    # calculate accuracy
    accuracy = (truePos + trueNeg)/(truePos + trueNeg + falsePos + falseNeg)
    return print(trainOrTest, "accuracy: %.2f" % accuracy)


"""
The following statements run the training and test perceptrons for the binary sets of data specified in Question 3.
Each is run for 20 iterations and with the L2 regularisation coefficient set to 0 so that it is not used at this stage.
Accuracies are calculated for each pair of data, both training and testing.
"""
# run binary perceptron for class 1 and class 2
finalWeights_1_2, predTrainList_1_2, trueTrainList_1_2 = perceptronTrain(trainData_1_2, trainLabel_1_2, 20, 0.0)
calcAccuracy("Class 1 and 2 - training", predTrainList_1_2, trueTrainList_1_2)
predTestList_1_2, trueTestList_1_2 = perceptronTest(finalWeights_1_2, testData_1_2, testLabel_1_2)
calcAccuracy("Class 1 and 2 - test", predTestList_1_2, trueTestList_1_2)

# run binary perceptron for class 2 and class 3
finalWeights_2_3, predTrainList_2_3, trueTrainList_2_3 = perceptronTrain(trainData_2_3, trainLabel_2_3, 20, 0.0)
calcAccuracy("Class 2 and 3 - training", predTrainList_2_3, trueTrainList_2_3)
predTestList_2_3, trueTestList_2_3 = perceptronTest(finalWeights_2_3, testData_2_3, testLabel_2_3)
calcAccuracy("Class 2 and 3 - test", predTestList_2_3, trueTestList_2_3)

# run binary perceptron for class 1 and class 3
finalWeights_1_3, predTrainList_1_3, trueTrainList_1_3 = perceptronTrain(trainData_1_3, trainLabel_1_3, 20, 0.0)
calcAccuracy("Class 1 and 3 - training", predTrainList_1_3, trueTrainList_1_3)
predTestList_1_3, trueTestList_1_3 = perceptronTest(finalWeights_1_3, testData_1_3, testLabel_1_3)
calcAccuracy("Class 1 and 3 - test", predTestList_1_3, trueTestList_1_3)


"""
The following statements run the training perceptron for the one-vs-rest data specified in Question 4.
Each is run for 20 iterations and initially with the L2 regularisation coefficient set to 0 so that it is not in affect at this stage.
"""
# create 3 models for multi class implementation. Last parameter is the L2 regularisation coefficient (set to 0.0 initially)
model_1, model_1_predTrainList, model_1_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_1, 20, 0.0)
model_2, model_2_predTrainList, model_2_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_2, 20, 0.0)
model_3, model_3_predTrainList, model_3_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_3, 20, 0.0)


"""
The following statements run the multi-class classifier which uses the above models.
The classifier is run once with the training data set and once with the test data set. 
Accuracies are reported for both.
"""
# run multi-class classifier with training data
multiPredTrain, multiTrueTrain = multiClassifier(model_1, model_2, model_3, multiTrainDataAll, multiTrainLabelAll)
calcAccuracy("Multi-class classifier - training data", multiPredTrain, multiTrueTrain)

# run multi-class classifier with test data
multiPredTest, multiTrueTest = multiClassifier(model_1, model_2, model_3, multiTestDataAll, multiTestLabelAll)
calcAccuracy("Multi-class classifier - test data", multiPredTest, multiTrueTest)


"""
The following statements run the training perceptron with the one-vs-rest data.
Each is run for 20 iterations and with the L2 regularisation coefficient starting at 0.01 and incrementing to 100 as specified in Question 5.
The multi-class classifier then uses those models and is run once with the training data set and once with the test data set. 
Accuracies are reported for both training data and test data for each of the 5 regularisation coefficient values.
"""
# run multi-class classifier with L2 regularisation coefficient of 0.01
model_1_001, model_1_001_predTrainList, model_1_001_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_1, 20, 0.01)
model_2_001, model_2_001_predTrainList, model_2_001_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_2, 20, 0.01)
model_3_001, model_3_001_predTrainList, model_3_001_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_3, 20, 0.01)
multiPredTrain_001, multiTrueTrain_001 = multiClassifier(model_1_001, model_2_001, model_3_001, multiTrainDataAll, multiTrainLabelAll)
calcAccuracy("Regularisation coefficient 0.01 - training data", multiPredTrain_001, multiTrueTrain_001)
multiPredTest_001, multiTrueTest_001 = multiClassifier(model_1_001, model_2_001, model_3_001, multiTestDataAll, multiTestLabelAll)
calcAccuracy("Regularisation coefficient 0.01 - test data", multiPredTest_001, multiTrueTest_001)

# run multi-class classifier with L2 regularisation coefficient of 0.1
model_1_01, model_1_01_predTrainList, model_1_01_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_1, 20, 0.1)
model_2_01, model_2_01_predTrainList, model_2_01_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_2, 20, 0.1)
model_3_01, model_3_01_predTrainList, model_3_01_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_3, 20, 0.1)
multiPredTrain_01, multiTrueTrain_01 = multiClassifier(model_1_01, model_2_01, model_3_01, multiTrainDataAll, multiTrainLabelAll)
calcAccuracy("Regularisation coefficient 0.1 - training data", multiPredTrain_01, multiTrueTrain_01)
multiPredTest_01, multiTrueTest_01 = multiClassifier(model_1_01, model_2_01, model_3_01, multiTestDataAll, multiTestLabelAll)
calcAccuracy("Regularisation coefficient 0.1 - test data", multiPredTest_01, multiTrueTest_01)

# run multi-class classifier with L2 regularisation coefficient of 1.0
model_1_1, model_1_1_predTrainList, model_1_1_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_1, 20, 1.0)
model_2_1, model_2_1_predTrainList, model_2_1_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_2, 20, 1.0)
model_3_1, model_3_1_predTrainList, model_3_1_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_3, 20, 1.0)
multiPredTrain_1, multiTrueTrain_1 = multiClassifier(model_1_1, model_2_1, model_3_1, multiTrainDataAll, multiTrainLabelAll)
calcAccuracy("Regularisation coefficient 1.0 - training data", multiPredTrain_1, multiTrueTrain_1)
multiPredTest_1, multiTrueTest_1 = multiClassifier(model_1_1, model_2_1, model_3_1, multiTestDataAll, multiTestLabelAll)
calcAccuracy("Regularisation coefficient 1.0 - test data", multiPredTest_1, multiTrueTest_1)

# run multi-class classifier with L2 regularisation coefficient of 10.0
model_1_10, model_1_10_predTrainList, model_1_10_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_1, 20, 10.0)
model_2_10, model_2_10_predTrainList, model_2_10_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_2, 20, 10.0)
model_3_10, model_3_10_predTrainList, model_3_10_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_3, 20, 10.0)
multiPredTrain_10, multiTrueTrain_10 = multiClassifier(model_1_10, model_2_10, model_3_10, multiTrainDataAll, multiTrainLabelAll)
calcAccuracy("Regularisation coefficient 10.0 - training data", multiPredTrain_10, multiTrueTrain_10)
multiPredTest_10, multiTrueTest_10 = multiClassifier(model_1_10, model_2_10, model_3_10, multiTestDataAll, multiTestLabelAll)
calcAccuracy("Regularisation coefficient 10.0 - test data", multiPredTest_10, multiTrueTest_10)

# run multi-class classifier with L2 regularisation coefficient of 100.0
model_1_100, model_1_100_predTrainList, model_1_100_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_1, 20, 100.0)
model_2_100, model_2_100_predTrainList, model_2_100_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_2, 20, 100.0)
model_3_100, model_3_100_predTrainList, model_3_100_trueTrainList = perceptronTrain(multiTrainDataAll, multiTrainLabel_3, 20, 100.0)
multiPredTrain_100, multiTrueTrain_100 = multiClassifier(model_1_100, model_2_100, model_3_100, multiTrainDataAll, multiTrainLabelAll)
calcAccuracy("Regularisation coefficient 100.0 - training data", multiPredTrain_100, multiTrueTrain_100)
multiPredTest_100, multiTrueTest_100 = multiClassifier(model_1_100, model_2_100, model_3_100, multiTestDataAll, multiTestLabelAll)
calcAccuracy("Regularisation coefficient 100.0 - test data", multiPredTest_100, multiTrueTest_100) 