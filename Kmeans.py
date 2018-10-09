import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# boolean solutionsVector

class Kmeans:
    def __init__(self, df, k, centroids = None, solutionsVector = None):
        self.solutionsVector = solutionsVector
        self.df = df
        self.k = k
        self.dimensions = df.shape[1]
        if centroids:
            self.centroids = centroids
        else:
            self.centroids = self.randomiseCentroids()
        self.classifiedDF = self.df.copy()
        self.plotIdx = 0
        self.colorPalette = sns.color_palette("hls", k)

    def randomiseCentroids(self):
        return [self.randomisedCentroid() for i in range(self.k)]

    def randomisedCentroid(self):
        return np.array([random.randint(int(self.df.iloc[:, j].min())+1, int(self.df.iloc[:, j].max()))
                         for j in range(self.dimensions)])

    def setCentroidsToGeographicalCenter(self):
        def findCentroid(df):
            return df.sum() / df.shape[0]

        gbIdx = self.classifiedDF.groupby('centroid')
        for i in range(self.k):
            try:
                groupDF = self.df.iloc[gbIdx.get_group(i).index]
            except:
                print('Empty Centroid: Reshuffleing')
                self.centroids[i] = self.randomisedCentroid()
                continue
            self.centroids[i] = np.array(findCentroid(groupDF))

    def findCentroid(self, datapoint):
        centroid = self.centroids[0]
        centroidIndex = 0
        for i in range(1, len(self.centroids)):
            if self.distanceOf(self.centroids[i], datapoint) < self.distanceOf(centroid, datapoint):
                centroid = self.centroids[i]
                centroidIndex = i
        return centroidIndex

    def distanceOf(self, a, b):
        return np.linalg.norm(a - b)

    def dissimilarity(self):
        total = 0
        for i, centroid in enumerate(self.centroids):
            points = self.classifiedDF.loc[self.classifiedDF.centroid == i]
            points = points.drop('centroid', axis = 1)
            for point in points.values:
                total += self.distanceOf(centroid, point) ** 2
        return total

    def calculate(self, maxIterations = 100, plotAll=False, showProgress=False):
        for i in range(maxIterations):
            self.classifiedDF['centroid'] = pd.Series(self.findCentroid(self.df.iloc[i]) for i in self.df.index)
            before = self.centroids.copy()
            self.setCentroidsToGeographicalCenter()
            if np.array_equal(before, self.centroids):
                return
            if plotAll:
                self.plot(show=False)
            if showProgress:
                print("gen {} done".format(i))

    def plot(self, show=False):
        plt.gcf().clear()
        gb = self.classifiedDF.groupby('centroid')
        for i in range(self.k):
            try:
                group = gb.get_group(i)
            except:
                print('WARNING: Empty group detected during plotting')
                continue
            plt.scatter(group.iloc[:,0], group.iloc[:,1], color = self.colorPalette[i])

        plt.scatter([x[0] for x in self.centroids], [x[1] for x in self.centroids], marker='x', color='red')
        if show:
            plt.show()
        else:
            plt.savefig('plots/{}'.format(self.plotIdx))
            self.plotIdx += 1

    def showPositiveFractions(self):
        classfiedWithSolutions = self.classifiedDF.copy()
        classfiedWithSolutions['solutionKmeans'] = self.solutionsVector
        for centroidID in range(self.k):
            classifiedSlice = classfiedWithSolutions.loc[classfiedWithSolutions.centroid == centroidID]
            positives = classifiedSlice.solutionKmeans.mean()
            print("Centroid ID: {}\nCentroid Coords:{}\nPositives: {}".format(centroidID, self.centroids[centroidID], positives))

    def trialsUntilMinimalDissimilarity(self, trials = 1, showProgress = False):
        assert trials >= 1, 'at least one trial required'
        if showProgress:
            print("trial: 1")
        self.calculate()
        bestDissimilarity = self.dissimilarity()
        bestCentroids = self.centroids.copy()
        bestClassification = self.classifiedDF.copy()

        for i in range(trials - 1):
            if showProgress:
                print("trial: {}".format(i))
            self = Kmeans(self.df, self.k)
            self.calculate()
            dissimilarity = self.dissimilarity()
            if dissimilarity < bestDissimilarity:
                bestDissimilarity = dissimilarity
                bestCentroids = self.centroids.copy()
                bestClassification = self.classifiedDF.copy()
        print('Randomised trials finished. Best dissimilarity = {}'.format(bestDissimilarity))
        self.centroids = bestCentroids
        self.classifiedDF = bestClassification

if __name__ == '__main__':
    kmeans = Kmeans(pd.read_csv('data/s1'), k = 15, solutionsVector=([False] * 4000 + [True] * 927))
    kmeans.calculate(maxIterations = 1)
    kmeans.showPositiveFractions()
    kmeans.plot(show=True)