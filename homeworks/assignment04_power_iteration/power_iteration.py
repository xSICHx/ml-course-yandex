import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    needed_matrix = np.eye(data.shape[0])
    tmp_data = data.copy()
    power = num_steps
    while power > 0:
        if power % 2 == 1:
            needed_matrix = needed_matrix @ tmp_data
            needed_matrix /= np.max(needed_matrix)
        tmp_data = tmp_data@tmp_data
        tmp_data /= np.max(tmp_data)
        power //= 2
        i += 1


    R = needed_matrix @ np.random.normal(loc = 0, scale = 1, size = needed_matrix.shape[0])
    eigenvector = R / np.linalg.norm(R)
    eigenvalue = np.mean((data @ eigenvector) / eigenvector)
    return float(eigenvalue), eigenvector 
    # pass