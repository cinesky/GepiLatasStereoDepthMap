import cv2
import numpy as np

window_size = 5

left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=32,
    blockSize=5,
    P1=8*3*window_size**2,
    P2=32*3*window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.StereoSGBM_MODE_HH4
)
lmbda = 80000
sigma = 1.0
visual_multipiler = 1.6

frameC = 1

while True:

    leftIs = "C:\\Users\Matyi\\Documents\\Depth_\\RealDataSets\\LeftJPG\\image"+str(frameC)+".jpg"
    rightIs = "C:\\Users\Matyi\\Documents\\Depth_\\RealDataSets\\RightJPG\\image"+str(frameC)+".jpg"

    left = cv2.imread(leftIs).astype(np.uint8)
    right = cv2.imread(rightIs).astype(np.uint8)
    frameC = frameC+1


    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(left, right)

    dispr = right_matcher.compute(right, left)
    
    displ = np.int16(displ)
    dispr = np.int16(dispr)

    filteredIMG = wls_filter.filter(displ, left, None, dispr)
    filteredIMG = cv2.normalize(src=filteredIMG, dst=filteredIMG, beta=0, alpha=255, norm_type = cv2.NORM_MINMAX)
    filteredIMG = cv2.bitwise_not(filteredIMG)
    cv2.imshow('DispartiyIMG', filteredIMG.astype(np.uint8))
    #cv2.imshow("Left", left)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


