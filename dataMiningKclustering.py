import numpy as np
import matplotlib.pyplot as plt



class ClusterResult:
    """
    Holds the output of the K-clustering algorithm and consequent evaluation calculations.
    
    Attributes:
        kvalue - (int) specified k value (number of clusters) tested.
        meanPrecision - (float) the calculated B-Cubed precision for the dataset.
        meanRecall - (float) the calculated B-Cubed recall for the dataset.
        meanF_score - (float) the calculated B-Cubed F-score for the dataset.

    """
    def __init__(self, kvalue, meanPrecision, meanRecall, meanF_score):
        self.kvalue = kvalue
        self.meanPrecision = meanPrecision
        self.meanRecall = meanRecall
        self.meanF_score = meanF_score


def load_data(file):
    """
    Processes the data.
    Initially removes the first non-numerical column then separates datapoints with commas.
    
    Parameters:
        file - (file) data file to be loaded.
    Returns:
        data - (ndarray) the processed data.
    """
    features = []
    with open(file) as f:
        for row in f:
            # split string after first element (ignore first column)
            row = row.split(' ',1)[1]

            # replace whitespace with commas and convert to array instead of string
            row = np.fromstring(row, sep=' ')
            features.append(row)
    return np.array(features)


animals = load_data("animals")
countries = load_data("countries")
fruits = load_data("fruits")
veggies = load_data("veggies")

#join all data together
dataset = np.concatenate((animals, countries, fruits, veggies))

# L2 normalise each vector
L2Norm = np.linalg.norm(dataset, axis=0, ord = 2)
datasetNorm = dataset/L2Norm


def initialCentroid(k, numDatapoints):
    """
    Picks a rando number of integers from a range equal to the length of the dataset.
    Number picked is equal to the number of clusters wanted (k-value).
    These are used as indexes to choose datapoints from the dataset to act as initial centroids.
    
    Parameters:
        k - (int) specified k value (number of clusters) tested.
        numDatapoints - (int) the length of the dataset.
    Returns:
        centroids  - (ndarray) datapoints to use as initial centroids.
    """
    # use seed so results are reproducible 
    np.random.seed(42)
    centroidIdx = np.random.choice(numDatapoints, k, replace=False)

    # add each indexes correesponding datapoint to a list
    centroids = []
    for i in centroidIdx:
        centroids.append(dataset[i])
        
    # convert centroids list to array
    centroids = np.array(centroids)
    return centroids


def sqL2Distance(datapoint, centroid):
    """
    Calculates the L2 (Euclidean) distance between a datapoint and a centroid
    
    Parameters:
        datapoint - (ndarray) single datapoint from dataset.
        centroid - (ndarray) single centroid.
    Returns:
        calculation - (float) L1 norm distance.
    """
    return np.square(np.linalg.norm(datapoint - centroid, ord=2))


def L1Distance(datapoint, centroid):
    """
    Calculates the L1 (Manhatten) distance between a datapoint and a centroid
    
    Parameters:
        datapoint - (ndarray) single datapoint from dataset.
        centroid - (ndarray) single centroid.
    Returns:
        calculation - (float) L1 norm distance.
    """
    return np.linalg.norm(datapoint - centroid, ord=1)


def assign(centroids, dataset, model):
    """
    Calculates the distance between each datapoint and centroid then assigns nearest centroid to datapoint.
    
    Parameters:
        centroids - (ndarray) datapoints to use as initial centroids.
        dataset - (ndarray) data to be split into clusters.
        model -(string) type of clustering model to use. Either k-means or k-medians.
    Returns:
        clusters  - (list) corresponding cluster assignment for each datapoint.
    """
    clusters = []
    for point in dataset:
        distance = []

        # use either L2 (Euclidean)distance or L1 (Manhatten) distance depending on model specified
        for centroid in centroids:
            if model == 'mean':
                distance.append(sqL2Distance(point,centroid))
            if model == 'median':
                distance.append(L1Distance(point,centroid))

        # choose centroid with minimum distance
        clusters.append(np.argmin(distance))
    return clusters


def initialCentroid(k, numDatapoints):
    """
    Picks a rando number of integers from a range equal to the length of the dataset.
    Number picked is equal to the number of clusters wanted (k-value).
    These are used as indexes to choose datapoints from the dataset to act as initial centroids.
    
    Parameters:
        k - (int) specified k value (number of clusters) tested.
        numDatapoints - (int) the length of the dataset.
    Returns:
        centroids  - (ndarray) datapoints to use as initial centroids.
    """
    # use seed so results are reproducible 
    np.random.seed(42)
    centroidIdx = np.random.choice(numDatapoints, k, replace=False)

    # add each indexes correesponding datapoint to a list
    centroids = []
    for i in centroidIdx:
        centroids.append(dataset[i])
        
    # convert centroids list to array
    centroids = np.array(centroids)

    return centroids


def k_cluster(k, dataset, iterations, model):
    """
    Takes a dataset and attempts to cluster the datapoints into a number of clusters equal to k.
    Uses either K-means or K-medians model depending on inputted parameters.
    
    Parameters:
        k - (int) specified k value (number of clusters) tested.
        dataset - (ndarray) data to be split into clusters.
        iterations -(int) maximum number of iterations through clustering in case of no convergence.
        model -(string) type of clustering model to use. Either k-means or k-medians.
    Returns:
        ClusterResult  - (class) contains the k value and mean precision, recall and F-score results for that value of k
    """
    # print model type and k in order to track progress of algorithm and separate evaluation print-outs
    print('(', model, ')  k value:',k)

    # create one-dimensional array of datapoints
    numDatapoints = len(dataset)

    # choose centroid indices randomly from dataset
    centroids = initialCentroid(k, numDatapoints)

    # make initial assignment of datapoints to clusters
    clusterList = assign(centroids, dataset, model)

    for _ in range(iterations):
        clusterList = assign(centroids, dataset, model)

        # compute mean or median (depending on model specified) of each cluster and set as new centroid
        newCentroids = []
        for cluster in range(k):
            if model == 'mean':
                newCentroids.append(np.mean([dataset[x] for x in range(len(dataset)) if clusterList[x] == cluster], axis=0))
            if model == 'median':
                newCentroids.append(np.median([dataset[x] for x in range(len(dataset)) if clusterList[x] == cluster], axis=0))
        
        # convert list to array to check for convergence
        newCentroids = np.array(newCentroids)
        if np.all(centroids == newCentroids):
            break

        # update centroids
        centroids = newCentroids

    # run evaluation method for this k value and model
    return evaluation(clusterList, k)


def evaluation(clusterList, k):
    """
    Calculates the B-Cubed precision, recall and F-score for a specified k value.
    
    Parameters:
        clusterList - (list) assigned cluster labels outputted by model
        k - (int) specified k value (number of clusters) tested
    Returns:
        ClusterResult  - (class) contains the k value and mean precision, recall and F-score results for that value of k
    """
    totalPrecision = []
    totalRecall = []
    totalF_score = []
    trueLabels = []

    # create dictionary for data, true labels and cluster
    dataDict = {}

    for idx, x in enumerate(dataset):
        dataDict[idx] = []
        # create list of true labels for evaluations
        if ([x] == animals).all(1).any():
            trueLabels.append('A')
        if ([x] == countries).all(1).any():
            trueLabels.append('B')
        if ([x] == fruits).all(1).any():
            trueLabels.append('C')
        if ([x] == veggies).all(1).any():
            trueLabels.append('D')

        # get truth label for datapoint
        A_x = trueLabels[idx]
        # get cluster label for datapoint
        C_x = clusterList[idx]

        # build dictionary of data. Index as key, cluster label and true label as values
        dataDict[idx].append(C_x)
        dataDict[idx].append(A_x)


        numSameCluster = 0
        numSameTrue = 0
        numSameClusterAndTrue = 0

        # check each dictionary entry to see if values match and add to counts if they do
        for key in dataDict:
            if dataDict[key][0] == C_x:
                numSameCluster +=1

            if dataDict[key][1] == A_x:
                numSameTrue +=1

            if dataDict[key][0] == C_x and dataDict[key][1] == A_x:
                numSameClusterAndTrue +=1

        # calculate B-cubed precision, recall and F-score for each datapoint
        precision = numSameClusterAndTrue/numSameCluster
        recall = numSameClusterAndTrue/numSameTrue
        f_score = (2 * recall * precision)/recall + precision

        # add values to lists to assemble values for whole dataset
        totalPrecision.append(precision)
        totalRecall.append(recall)
        totalF_score.append(f_score)

    # calculate mean values for dataset
    meanPrecision = np.mean(totalPrecision)
    meanRecall = np.mean(totalRecall)
    meanF_score = np.nanmean(totalF_score)

    print('mean precision %.2f' % meanPrecision)
    print('mean recall %.2f' % meanRecall)
    print('mean F-score %.2f' % meanF_score)

    return ClusterResult(k, meanPrecision, meanRecall, meanF_score)


def plotEvals(data, title):
    """
    Plots a scatter graph of the B-Cubed precision, recall and F-score (y-axis) for each K-value (x-axis).
    
    Parameters:
        data - (list) the results of the evaluation function
        title - (string) model type and "Normalised" if data was normalised.
    Returns:
        None
    """
    fig,ax = plt.subplots()
    fig.set_size_inches(5, 5)
    ax.set_title(title)

    for val in data:
        ax.scatter(val.kvalue, val.meanPrecision, c='tab:red', marker='X')
        ax.scatter(val.kvalue, val.meanRecall, c='tab:blue', marker='^')
        ax.scatter(val.kvalue, val.meanF_score, c='tab:orange', marker='o')

    #plt.title('Evaluations')
    plt.xlabel('K Value')
    plt.ylim([0, 3])
    fig.tight_layout()
    ax.legend(['Precision', 'Recall', 'F-score'], loc="center right")
    plt.show()


# run each of the sets of data and k values from 1 to 9 for each question in the assignment
resultsMean = []
for k in range(1,10):
    resultsMean.append(k_cluster(k, dataset, 20, 'mean'))
plotEvals(resultsMean, 'K-Means')

resultsMeanNorm = []
for k in range(1,10):
    resultsMeanNorm.append(k_cluster(k, datasetNorm, 20, 'mean'))
plotEvals(resultsMeanNorm, 'K-Means Normalised')

resultsMedian = []
for k in range(1,10):
    resultsMedian.append(k_cluster(k, dataset, 20, 'median'))
plotEvals(resultsMedian, 'K-Medians')

resultsMedianNorm = []
for k in range(1,10):
    resultsMedianNorm.append(k_cluster(k, datasetNorm, 20, 'median'))
plotEvals(resultsMedian, 'K-Medians Normalised')
