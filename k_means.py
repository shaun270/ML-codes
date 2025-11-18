import numpy as np

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
	# step 1 - assign random points to a dataset
    cluster_points = np.array(initial_centroids)
    points = np.array(points)
    # print(points)
    # print(np.array(points).shape[0])
    # step 2 - calculate distance of each point from centroids and assign
    for j in range(max_iterations):
        indexes = []
        for i in range(points.shape[0]):
            # print(np.linalg.norm(cluster_points - np.array(points[i]), axis = 1))
            distances = np.linalg.norm(cluster_points - points[i], axis = 1)
            # print(f"distances : {distances}")
            indexes.append(np.argmin(distances))
        print(indexes)
        # break
        # step 3 - find the centroid for each point
        for i in range(k):
            # print(f"index {indexes} k {k}")
            # print((indexes)==k)
            boolean_array = [ x == i for x in indexes]
            # print(points[(indexes == i)])
            cluster = points[boolean_array]

            # print(cluster)
            # if cluster
            # print(cluster)
            cluster_points[i] = np.mean(cluster, axis = 0)
   
    rounded = np.round(cluster_points, 4)
    final_centroids = [tuple(i) for i in rounded]
    return final_centroids

print(k_means_clustering(points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], k = 2, initial_centroids = [(1, 1), (10, 1)], max_iterations = 10))