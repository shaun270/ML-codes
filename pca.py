import numpy as np

def pca(data, k):
    # Standardize the data
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Compute the covariance matrix
    covariance_matrix = np.cov(data_standardized, rowvar=False)

    # Eigen decomposition using eigh (for symmetric matrices)
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort by descending eigenvalues (reverse the order from eigh)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, idx]

    # Select the top k eigenvectors (principal components)
    principal_components = eigenvectors_sorted[:, :k]

    # Fix direction (sign): eigenvectors are only defined up to +/-,
    # so we make the first "significant" element positive for consistency
    for i in range(k):
        for j in range(principal_components.shape[0]):
            if np.abs(principal_components[j, i]) > 1e-10:
                if principal_components[j, i] < 0:
                    principal_components[:, i] *= -1
                break

    return np.round(principal_components.real, 4)