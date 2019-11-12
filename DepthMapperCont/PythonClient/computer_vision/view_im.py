import airsim #pip install airsim
import cv2
import numpy as np

# for car use CarClient() 
client = airsim.CarClient()

png_image = client.simGetImage("0", airsim.ImageType.Scene)

cv2.imshow(png_image)
cv2.waitKey(0)
cv2.destroyAllWindows()