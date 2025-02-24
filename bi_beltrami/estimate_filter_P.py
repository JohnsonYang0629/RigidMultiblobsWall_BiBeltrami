import random
import numpy as np
import math

def sphere_P0Cheat_v2(theta, x, am):
    """
    Compute the analytic projection matrix and orthonormal tangent vector.

    Inputs:
    - theta: intrinsic coordinates.
    - x: extrinsic coordinates.
    - am: radius of sphere.

    Returns:
    - projection matrix P0. 
    - tangent vector tvec2.
    """
    # projection matrix P0 and tangent vector tvec2
    N = x.shape[0]
    n = x.shape[1]
    d = theta.shape[1]

    THETA = theta[:, 0]
    PHI = theta[:, 1]

    # tangent direction
    tvec = np.zeros((d, n, N))
    # tangent 1
    tvec[0, 0, :] = am * np.array([math.cos(tt) for tt in THETA]) * np.array([math.cos(pp) for pp in PHI])
    tvec[0, 1, :] = am * np.array([math.cos(tt) for tt in THETA]) * np.array([math.sin(pp) for pp in PHI])
    tvec[0, 2, :] = - am * np.array([math.sin(tt) for tt in THETA])
    # tangent 2
    tvec[1, 0, :] = - am * np.array([math.sin(tt) for tt in THETA]) * np.array([math.sin(pp) for pp in PHI])
    tvec[1, 1, :] = am * np.array([math.sin(tt) for tt in THETA]) * np.array([math.cos(pp) for pp in PHI])
    tvec[1, 2, :] = np.array([math.cos(tt) for tt in THETA]) * 0

    # normalized tangent direction
    tvec2 = np.zeros((d, n, N))
    # normalized tangent 1
    tvec2[0, :, :] = tvec[0, :, :] / am
    # tangent 2 permute(tvec2,[3,2,1]);
    tvec2_len = np.sqrt(np.squeeze(tvec[1, 0, :] ** 2 + tvec[1, 1, :] ** 2 + tvec[1, 2, :] ** 2))
    tvec2[1, 0, :] = tvec[1, 0, :] / tvec2_len
    tvec2[1, 1, :] = tvec[1, 1, :] / tvec2_len
    tvec2[1, 2, :] = tvec[1, 2, :] / tvec2_len

    # analytic true projection matrix 
    P0 = np.zeros((n, n, N))
    for ii in range(N):
        P0[:, :, ii] = tvec[:, :, ii].T @ np.linalg.pinv(tvec[:, :, ii] @ tvec[:, :, ii].T) @ tvec[:, :, ii]
    P0 = P0.transpose((2, 1, 0)) # P0 is N*n*n
    return P0, tvec2

