import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from GeographicDistances.Point import Point


class Kmeans:
    def __init__(self, df=pd.read_csv('data/s1'), k=15, country_column=None):
        self.df = df.reset_index(drop=True)
        self.train_columns = df.columns
        self.k = k
        self.dimensions = df.shape[1]
        self.country_column = country_column
        if country_column:
            self.country_index = list(self.df.columns.values).index(self.country_column)
            self.convert_country_column()
        self.centroids = self.randomise_centroids()
        self.classified_df = self.df.copy()
        self.plot_idx = 0
        self.color_palette = sns.color_palette("hls", k)

    def convert_country_column(self):
        self.df[self.country_column] = self.df[self.country_column].apply(lambda code: Point(code))

    def randomise_centroids(self):
        return [self.randomise_centroid() for i in range(self.k)]

    def randomise_centroid(self):
        res = []
        for j in range(self.dimensions):
            if j == self.country_index:
                res.append(Point(randomise=True))
            else:
                res.append(
                    0.01 * random.randint(int(100 * self.df.iloc[:, j].min()), int(100 * self.df.iloc[:, j].max())))
        return np.array(res)

    def set_mean_to_centroids(self):
        def find_centroid(group_df):
            non_country_columns = self.df.columns.drop(self.country_column)
            res = group_df[non_country_columns].sum() / group_df.shape[0]

            def find_geo_centroid(group_df):
                total_lat = 0
                total_lon = 0
                for point in group_df[self.country_column]:
                    total_lat += point.lat
                    total_lon += point.lon
                return Point(lat=total_lat / group_df.shape[0], lon=total_lon / group_df.shape[0])

            if self.country_column:
                res[self.country_column] = find_geo_centroid(group_df)
            return res[self.train_columns]

        gb_idx = self.classified_df.groupby('centroid')
        for i in range(self.k):
            if i not in self.classified_df.centroid.unique():
                print('Empty Centroid: Reshuffling')
                self.centroids[i] = self.randomise_centroid()
            else:
                group_df = self.df.loc[gb_idx.get_group(i).index]
                self.centroids[i] = np.array(find_centroid(group_df))

    def find_centroid(self, datapoint):
        datapoint = np.array(datapoint)
        centroid = self.centroids[0]
        centroid_index = 0
        for i in range(1, len(self.centroids)):
            if self.distance_of(self.centroids[i], datapoint) < self.distance_of(centroid, datapoint):
                centroid = self.centroids[i]
                centroid_index = i
        return centroid_index

    def distance_of(self, a, b):
        non_country_columns = self.df.columns.drop(self.country_column)
        euklidian_distances = 0
        if not non_country_columns.empty:
            euklidian_distances += np.linalg.norm(np.delete(a, self.country_index)
                                                  - np.delete(b, self.country_index))

        return euklidian_distances + a[self.country_index].norm_dist(b[self.country_index])

    def calculate(self, iterations=1, plot_all=False):
        for iteration in range(iterations):
            self.classified_df['centroid'] = self.df.apply(self.find_centroid, axis=1)
            self.set_mean_to_centroids()
            if plot_all:
                self.plot(show=False)

    def plot(self, show=False):
        """
        Only applicable to 2D (plottable) datasets
        :param show:
        :return:
        """
        plt.gcf().clear()
        gb = self.classified_df.groupby('centroid')
        for i in range(self.k):
            try:
                group = gb.get_group(i)
            except:
                print('WARNING: Empty group detected during plotting')
                continue
            plt.scatter(group.x, group.y, color=self.color_palette[i])

        plt.scatter([x[0] for x in self.centroids], [x[1] for x in self.centroids], marker='x', color='red')
        if show:
            plt.show()
        else:
            plt.savefig('plots/{}'.format(self.plot_idx))
            self.plot_idx += 1

    def dissimilarity(self):
        total = 0
        for i, centroid in enumerate(self.centroids):
            points = self.classified_df.loc[self.classified_df.centroid == i]
            points = points.drop('centroid', axis=1)
            for point in points.values:
                total += self.distance_of(centroid, point) ** 2
        return total

    def trials_until_minimal_dissimilarity(self, trials=1, show_progress=False):
        assert trials >= 1, 'at least one trial required'
        if show_progress:
            print("trial: 1")
        self.calculate()
        best_dissimilarity = self.dissimilarity()
        best_centroids = self.centroids.copy()
        best_classification = self.classified_df.copy()

        for i in range(trials - 1):
            if show_progress:
                print("trial: {}".format(i))
            self.__init__(self.df, self.k)
            self.calculate()
            dissimilarity = self.dissimilarity()
            if dissimilarity < best_dissimilarity:
                best_dissimilarity = dissimilarity
                best_centroids = self.centroids.copy()
                best_classification = self.classified_df.copy()
        print('Randomised trials finished. Best dissimilarity = {}'.format(best_dissimilarity))
        self.centroids = best_centroids
        self.classified_df = best_classification

    def __repr__(self):
        return self.classified_df.__repr__()


if __name__ == '__main__':
    df = pd.DataFrame(pd.read_msgpack('GeographicDistances/geography').index)
    df.columns = ['country']
    kmeans = Kmeans(df, k=5, country_column='country')
    kmeans.calculate(iterations=2)
    print(kmeans.classified_df.sort_values('centroid'))
