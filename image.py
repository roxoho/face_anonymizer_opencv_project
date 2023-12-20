import cv2, os
import mediapipe as mp
#read image
img = cv2.imread(os.path.join('.', 'image.jpg'))

H, W,_ = img.shape
#detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(min_detection_confidence=0.5,model_selection=0) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections is not None:
        for detect in results.detections:
            location_Data =detect.location_data
            bbox = location_Data.relative_bounding_box
    
            x1,y1,w,h= bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1,y1,w,h = int(x1*W), int(y1*H), int(w*W), int(h*H)

            #cv2.rectangle(img, (x1,y1), (x1+w,y1+h), (0,255,0), 5)

            #blure face
            img[y1:y1+h, x1:x1+w,:] = cv2.blur(img[y1:y1+h, x1:x1+w,:], (50,50))
    
    #cv2.imshow('img', img)
    #cv2.waitKey(0)




#save image
cv2.imwrite(os.path.join('.', 'image_blured.jpg'), img)