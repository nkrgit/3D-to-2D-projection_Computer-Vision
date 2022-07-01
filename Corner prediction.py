###############
##Design the function "calibrate" to  return
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates.
#                                            True if the intrinsic parameters are invariable.
# It is ok to add other functions if you need
###############


import numpy as np
import cv2
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):
    w = np.array(
        [[40, 0, 40, 1], [40, 0, 30, 1], [40, 0, 20, 1], [40, 0, 10, 1], [30, 0, 40, 1], [30, 0, 30, 1], [30, 0, 20, 1],
         [30, 0, 10, 1], [20, 0, 40, 1], [20, 0, 30, 1], [20, 0, 20, 1], [20, 0, 10, 1], [10, 0, 40, 1], [10, 0, 30, 1],
         [10, 0, 20, 1], [10, 0, 10, 1], [0, 0, 40, 1], [0, 0, 30, 1], [0, 0, 20, 1], [0, 0, 10, 1], [0, 10, 40, 1],
         [0, 10, 30, 1], [0, 10, 20, 1], [0, 10, 10, 1], [0, 20, 40, 1], [0, 20, 30, 1], [0, 20, 20, 1], [0, 20, 10, 1],
         [0, 30, 40, 1], [0, 30, 30, 1], [0, 30, 20, 1], [0, 30, 10, 1], [0, 40, 40, 1], [0, 40, 30, 1], [0, 40, 20, 1],
         [0, 40, 10, 1]])

    chessboardSize = (4, 9)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria

    img = cv2.imread(imgname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)  # chess board corners

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)  # Draw and display the corners

    # World Coordinates
    count = 0
    k0 = [0, 0, 0, 0]
    A = np.zeros((72, 12))
    p = np.zeros((3, 4))
    p0 = np.zeros((1, 12))
    x1 = np.zeros((1, 3))
    x2 = np.zeros((1, 3))
    x3 = np.zeros((1, 3))
    m1 = np.zeros((1, 3))
    m2 = np.zeros((1, 3))
    m3 = np.zeros((1, 3))

    for i in range(0, 71, 2):
        j = i + 1

        k1 = w[count]
        k2 = -corners2[count][0][0] * w[count]

        ki1 = np.append(k1, k0)
        A[i] = np.append(ki1, k2)

        k3 = -corners2[count][0][1] * w[count]

        ki2 = np.append(k0, k1)
        A[j] = np.append(ki2, k3)

        count = count + 1

    [u, e, vt] = np.linalg.svd(A)

    p0 = vt[11, :]
    p = np.reshape(p0, (3, 4))

    for i in range(3):
        x1[0][i] = p[0][i]
        x2[0][i] = p[1][i]
        x3[0][i] = p[2][i]

    s = np.sum(pow(x3, 2))
    lam = np.sqrt(1 / s)  # From rotation matrix equation r31^2+r32^2+r33^2 = 1
    #print(lam)

    m1 = lam * x1
    m2 = lam * x2
    m3 = lam * x3
    ox = np.dot(m1, m3.T)
    oy = np.dot(m2, m3.T)

    fx = np.sqrt(np.dot(m1, m1.T) - pow(ox, 2))
    fy = np.sqrt(np.dot(m2, m2.T) - pow(oy, 2))

    intrinsic_params = [fx[0][0], fy[0][0], ox[0][0], oy[0][0]]
    is_constant = True  # Remains same: There is a very slight difference which can be accounted for precision difference

    return intrinsic_params, is_constant


if __name__ == "__main__":

    '''w0 = np.array(
        [[40, 0, 30, 1], [40, 0, 20, 1], [40, 0, 10, 1], [40, 0, 0, 1], [30, 0, 30, 1], [30, 0, 20, 1], [30, 0, 10, 1],
         [30, 0, 0, 1], [20, 0, 30, 1], [20, 0, 20, 1], [20, 0, 10, 1], [20, 0, 0, 1], [10, 0, 30, 1], [10, 0, 20, 1],
         [10, 0, 10, 1], [10, 0, 0, 1], [0, 0, 30, 1], [0, 0, 20, 1], [0, 0, 10, 1], [0, 0, 0, 1], [0, 10, 30, 1],
         [0, 10, 20, 1], [0, 10, 10, 1], [0, 10, 0, 1], [0, 20, 30, 1], [0, 20, 20, 1], [0, 20, 10, 1], [0, 20, 0, 1],
         [0, 30, 30, 1], [0, 30, 20, 1], [0, 30, 10, 1], [0, 30, 0, 1], [0, 40, 30, 1], [0, 40, 20, 1], [0, 40, 10, 1],
         [0, 40, 0, 1]])'''  # Different world coordinates used for testing the difference in intrinsic parameters

    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)






