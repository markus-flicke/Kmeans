import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, df = pd.read_csv('data/s1'), k = 15):
        self.df = df
        self.k = k
        self.dimensions = df.shape[1]
        self.means = self.randomiseMeans()
        self.closestMeans = None
        self.classifiedDF = self.df.copy()
        self.plotIdx = 0

    def randomiseMeans(self):
        return [self.randomisedMean() for i in range(self.k)]

    def randomisedMean(self):
        return np.array([random.randint(self.df.iloc[:,j].min(), self.df.iloc[:,j].max())
                          for j in range(self.dimensions)])

    def setMeanToCentroids(self):
        def findCentroid(df):
            return df.sum() / df.shape[0]

        gbIdx = self.classifiedDF.groupby('closestMean')
        for i in range(self.k):
            try:
                groupDF = self.df.iloc[gbIdx.get_group(i).index]
            except:
                print('Empty Mean: Reshuffleing meansIndex')
                self.means[i] = self.randomisedMean()
                continue
            self.means[i] = np.array(findCentroid(groupDF))

    def findClosestMean(self, datapoint):
        closestMean = self.means[0]
        closestMeanIndex = 0
        for i in range(1, len(self.means)):
            if self.distanceOf(self.means[i], datapoint) < self.distanceOf(closestMean, datapoint):
                closestMean = self.means[i]
                closestMeanIndex = i
        return closestMeanIndex

    def distanceOf(self, a, b):
        return np.linalg.norm(a-b)

    def calculate(self, iterations = 1, plotAll = False):
        for i in range(iterations):
            self.classifiedDF['closestMean'] = pd.Series(self.findClosestMean(self.df.iloc[i]) for i in self.df.index)
            self.setMeanToCentroids()
            if plotAll:
                self.plot(show = False)

    def plot(self, show = False):
        plt.gcf().clear()
        gb = self.classifiedDF.groupby('closestMean')
        for i in range(self.k):
            try:
                group = gb.get_group(i)
            except:
                print('WARNING: Empty group detected during plotting')
                continue
            plt.scatter(group.x, group.y)

        plt.scatter([x[0] for x in self.means], [x[1] for x in self.means], marker='x', color='red')
        if show:
            plt.show()
        else:
            plt.savefig('plots/{}'.format(self.plotIdx))
            self.plotIdx += 1


if __name__ == '__main__':
    kmeans = Kmeans()
    kmeans.calculate(iterations = 5, plotAll=True)

