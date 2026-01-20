import numpy as np

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    cluster_points = np.array(initial_centroids)
    points = np.array(points)
    for j in range(max_iterations):
        indexes = []
        for i in range(points.shape[0]):
            distances = np.linalg.norm(cluster_points - points[i], axis = 1)
            indexes.append(np.argmin(distances))
        for i in range(k):
            boolean_array = [ x == i for x in indexes]
            cluster = points[boolean_array]
            cluster_points[i] = np.mean(cluster, axis = 0)
   
    rounded = np.round(cluster_points, 4)
    final_centroids = [tuple(i) for i in rounded]
    return final_centroids

