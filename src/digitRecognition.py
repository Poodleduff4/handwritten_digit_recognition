import numpy as np
import cv2
from skimage import img_as_ubyte		
from skimage.color import rgb2gray
from keras.models import load_model

width = 640
height = 480
cameraNo = 0
 
cap = cv2.VideoCapture(cameraNo)
cap.set(3,width)
cap.set(4,height)
 

model = load_model('trained_model.h5')
 
while True:
	success, img_original = cap.read()
	img_gray = rgb2gray(img_original)


	img_gray_u8 = img_as_ubyte(img_gray)
	#cv2.imshow("Window", img_gray_u8)
	
	#Convert grayscale image to binary
	(thresh, im_binary) = cv2.threshold(img_gray_u8, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	

	img_resized = cv2.resize(im_binary,(28,28))

	#invert image
	im_gray_invert = 255 - img_resized
	cv2.imshow("invert image", im_gray_invert)

	im_final = im_gray_invert.reshape(1,28,28,1)



	ans = model.predict(im_final)
	ans = np.argmax(ans,axis=1)[0]
	print(ans)


	cv2.putText(img_original,'Predicted Digit : '+str(ans),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

	cv2.imshow("Original Image",img_original)
	if cv2.waitKey(1) and 0xFF == ord('q'):
		break
        
cap.release()
cv2.destroyAllWindows()

'''
Dataset subplots

(x_train, y_train),(x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_train[i]),transform=ax.transAxes, color='green')
plt.show()

'''