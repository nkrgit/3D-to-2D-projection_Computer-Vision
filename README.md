# Corner Prediction_Computer-Vision
<ul>I calculated the projection matrix from world coordinate to image coordinate based on the 32 marked points on the checkerboard.</ul>

<ul>I have used OpenCV function cv2.findChessboardCorners to get the image coordinate of the 32 points (xn, yn), where n = 1, . . . , 32 and manually labelled the world coordinate of these points on checkboard (Xn, Yn, Zn).</ul>

<ul>Build the homogeneous linear equation Ax = 0 and solved the equation and got the solution x where |x| = 1. I solved for λ where m = λ · x </ul>

<ul>Finally solved for intrinsic parameters fx, fy, ox, oy</ul>

<h4> Note: Descripition is added assuming reader has a basic understanding of computer vision concepts </h4>
