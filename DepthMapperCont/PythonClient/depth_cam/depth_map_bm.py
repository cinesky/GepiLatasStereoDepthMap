import airsim
import cv2
import numpy as np


client = airsim.MultirotorClient()

IMAGE_NORMAL = airsim.ImageType.Scene

min_disp = 16
window_size = 17


stereo = cv2.StereoBM_create(numDisparities=32, blockSize=5)
stereo.setMinDisparity(0)
stereo.setNumDisparities(32)
stereo.setBlockSize(6)
stereo.setDisp12MaxDiff(1)
stereo.setUniquenessRatio(10)
stereo.setSpeckleRange(63)
stereo.setSpeckleWindowSize(5)

while(True):
    responsend_l = client.simGetImage("Front Left",IMAGE_NORMAL) # Unity error: camera named Front Left
    responsend_r = client.simGetImage("1", IMAGE_NORMAL)

    np_response_image_r = np.asarray(bytearray(responsend_r), dtype="uint8")
    np_response_image_l = np.asarray(bytearray(responsend_l), dtype="uint8")
    decoded_frame_r = cv2.imdecode(np_response_image_r, cv2.IMREAD_COLOR)
    decoded_frame_l = cv2.imdecode(np_response_image_l, cv2.IMREAD_COLOR)
    frameL_gray = cv2.cvtColor(decoded_frame_l, cv2.COLOR_BGR2GRAY)
    frameR_gray = cv2.cvtColor(decoded_frame_r, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(frameL_gray, frameR_gray).astype(np.uint8)
    filteredIMG = cv2.normalize(filteredIMG, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imshow("DisparityMap", filterIMG.astype(np.uint8))

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
client.reset() #for testing

