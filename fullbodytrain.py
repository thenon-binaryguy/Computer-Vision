import os
import cv2
from PIL import Image,ImageFilter
import numpy as np
import pickle
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "pedes")

face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels = []
x_train = []
for root, dirs, files in os.walk(image_dir):
        for file in files:
                #print("x")
                if file.endswith("png") or file.endswith("jpeg"):
                        path = os.path.join(root, file)
                        label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                        #print(label, path)
                        if label in label_ids:
                                pass
                        else:
                                label_ids[label]=current_id
                                current_id+=1
                                _id=label_ids[label]
                        print(label_ids)
                        #y_labels.append(label)
                        #x_train.append(path)
                        pil_image = Image.open(path).convert("L")
                        qil_image=pil_image.filter(ImageFilter.SMOOTH)
                        size = (550, 550)
                        final_image = qil_image.resize(size, Image.ANTIALIAS)
                        image_array = np.array(final_image, "uint8")
                        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=2)
                        for (x,y,w,h) in faces:
                                roi = image_array[y:y+h, x:x+w]
                                x_train.append(roi)
                                y_labels.append(_id)

print(y_labels)
print(x_train)
with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("fullbody.yml")
			
			
