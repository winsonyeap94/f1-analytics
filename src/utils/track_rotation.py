import numpy as np


def rotate(xy, *, angle):
    """
    Rotates a 2D vector by a given angle.

    Parameters:
    xy (numpy.ndarray): A 2D vector represented as a 1x2 numpy array.
    angle (float): The angle in radians by which to rotate the vector.

    Returns:
    numpy.ndarray: The rotated vector as a 1x2 numpy array.
    """
    rot_mat = np.array(
        [[np.cos(angle), np.sin(angle)],
         [-np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(xy, rot_mat)