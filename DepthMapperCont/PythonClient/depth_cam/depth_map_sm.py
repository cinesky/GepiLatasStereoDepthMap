import airsim
import cv2
import numpy as np

CAMERA_NAME = "0"
IMAGE_SEGMENT = airsim.ImageType.Segmentation
IMAGE_DEPTH = airsim.ImageType.DepthPerspective
IMAGE_NORMAL = airsim.ImageType.Scene
RIGHT_CAM = "front_right"
LEFT_CAM = "front_left"

DECODE_EXTEND = '.jpg'

client = airsim.MultirotorClient()

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

while(True):
    responsend_l = client.simGetImage("Front Left",IMAGE_NORMAL) # Unity error: camera named Front Left
    responsend_r = client.simGetImage("1", IMAGE_NORMAL)

    np_response_image_r = np.asarray(bytearray(responsend_r), dtype="uint8")
    np_response_image_l = np.asarray(bytearray(responsend_l), dtype="uint8")
    decoded_frame_r = cv2.imdecode(np_response_image_r, cv2.IMREAD_COLOR)
    decoded_frame_l = cv2.imdecode(np_response_image_l, cv2.IMREAD_COLOR)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    

    # computing disparity
    displ = left_matcher.compute(decoded_frame_l, decoded_frame_r)

    dispr = right_matcher.compute(decoded_frame_r, decoded_frame_l)
    displ = np.int16(displ)
    dispr = np.int16(dispr)

    filteredIMG = wls_filter.filter(displ, decoded_frame_l, None, dispr)

    filteredIMG = cv2.normalize(src=filteredIMG, dst=filteredIMG, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredIMG = cv2.bitwise_not(filteredIMG)
    cv2.imshow('DispartiyIMG', filteredIMG.astype(np.uint8))

    #cv2.imshow("Right", decoded_frame_r)
    cv2.imshow("Left", decoded_frame_l)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
client.reset() # for testing


