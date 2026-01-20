import numpy as np

def k_nearest_neighbors(points, query_point, k):
    points = np.array(points)
    query_point = np.array(query_point)

    
    distances = np.linalg.norm(points - query_point, axis = 1)
    x = np.argsort(distances)

    answers = []
    for i in x[:k]:
        answers.append(tuple(points[i]))

    return answers

