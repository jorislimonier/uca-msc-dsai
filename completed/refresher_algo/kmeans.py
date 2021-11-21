import numpy as np
import plotly.graph_objects as go

# np.random.seed(42)
nb_points = 1000
xdata1 = np.random.normal(0, 2, int(nb_points/2))
ydata1 = np.random.normal(2, 3, int(nb_points/2))
xdata2 = np.random.normal(5, 2, int(nb_points/2))
ydata2 = np.random.normal(19, 1, int(nb_points/2))

xdata = np.append(xdata1, xdata2)
ydata = np.append(ydata1, ydata2)
data = [[xdata[i], ydata[i]] for i in range(nb_points)]

class KMeans():
    def __init__(self, data, k=2, nb_points=nb_points):
        self.k = k
        self.nb_points = nb_points
        self.data = data
        self.centroids = np.array([self.data[i] for i in np.random.choice(range(self.nb_points), k)])
        

    def closest_centroid(self, point):
        centroid_distances = [np.linalg.norm(
            centroid - point) for centroid in self.centroids]
        return self.centroids[np.argmin(centroid_distances)]

    def predict(self):
        self.labels = np.zeros(len(self.data)) - 1
        # print(self.labels)
        for i, point in enumerate(self.data):
            # print(i, point)
            distances = [np.linalg.norm(np.array(point) - centroid) for centroid in self.centroids]
            self.labels[i] = int(np.argmin(distances))
        # return self.labels

    def plot_points(self):
        fig = go.Figure()
        x = self.centroids[:, 0]
        y = self.centroids[:, 1]
        scat = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            opacity=1,
            marker=dict(color="Black", size=20),
            name="Centroids")
        fig.add_trace(scat)
        for centroid_i in range(self.k):
            x = [self.data[i][0] for i in range(self.nb_points) if self.labels[i] == centroid_i]
            y = [self.data[i][1] for i in range(self.nb_points) if self.labels[i] == centroid_i]
            scat = go.Scatter(x=x, y=y, mode="markers", opacity=0.6, name=f"Cluster {centroid_i}")
            fig.add_trace(scat)
        return fig

    def relocate_centroids(self):
        for centroid_i in range(self.k):
            x = np.mean([self.data[i][0] for i in range(self.nb_points) if self.labels[i] == centroid_i])
            y = np.mean([self.data[i][1] for i in range(self.nb_points) if self.labels[i] == centroid_i])
            print(x)
            print(y)
            self.centroids[centroid_i] = [x, y]

km = KMeans(data)
km.centroids
km.predict()
km.plot_points().show()
km.relocate_centroids()