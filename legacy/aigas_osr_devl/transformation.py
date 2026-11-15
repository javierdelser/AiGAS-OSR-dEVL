#!/usr/bin/python

import numpy as np

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def get_rotation_translation(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < 0, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    # CAMBIO 5/4/23
    return R, t, centroid_A, centroid_B #centroid_A=Centroide de los SIN_TRANSFORMAR y  centroid_B= Centroide de los TRANSFORMADOS
def get_rotation_translation_MULTI_D(dim,A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    num_rows, num_cols = B.shape

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < 0, reflection detected!, correcting for it ...")
        Vt[(dim-1),:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    # CAMBIO 5/4/23
    return R, t, centroid_A, centroid_B #centroid_A=Centroide de los SIN_TRANSFORMAR y  centroid_B= Centroide de los TRANSFORMADOS

#MARIA: 5/4/23
def rotateMatrix(beta,sense):
    #angulo retacion: beta
    #sense retacion: sentido:POSITIVO,NEGATIVO,IDENTIDAD
    if sense=='POSITIVE':
        mat=np.array([[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]])
    elif sense=='NEGATIVE':
        mat=np.array([[np.cos(beta), np.sin(beta)], [-np.sin(beta), np.cos(beta)]])
    else:
        mat=np.array([[1, 0], [0, 1]])
    return mat
def point_to_point(origin_array,ang_rot,sense,centroide):
    '''
    https://stackoverflow.com/questions/74519927/best-way-to-rotate-and-translate-a-set-of-points-in-python

    Esta funcion sólo esta disponibe para movimientos en 2D
    origin_array: origin points [nx2 array]
    '''
    newxy1 = origin_array @ rotateMatrix(ang_rot,sense).T
    newxy2 = (origin_array - centroide) @ rotateMatrix(ang_rot,sense).T + centroide

    return newxy1,newxy2
