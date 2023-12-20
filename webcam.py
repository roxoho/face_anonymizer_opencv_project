import cv2, os
import mediapipe as mp
#read image
import argparse

def process_img(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    H, W,_ = img.shape
        

    if results.detections is not None:
        for detect in results.detections:
            location_Data =detect.location_data
            bbox = location_Data.relative_bounding_box

    
            x1,y1,w,h= bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1,y1,w,h = int(x1*W), int(y1*H), int(w*W), int(h*H)

            #cv2.rectangle(img, (x1,y1), (x1+w,y1+h), (0,255,0), 5)

            #blure face
            img[y1:y1+h, x1:x1+w,:] = cv2.blur(img[y1:y1+h, x1:x1+w,:], (50,50))
    
    return img
    

args = argparse.ArgumentParser()

args.add_argument("--mode", default="webcam", help="mode: image or video")
args.add_argument("--filePath", default="./a.mp4", help="path to image or video")

args = args.parse_args()



#detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(min_detection_confidence=0.5,model_selection=0) as face_detection:

    if args.mode in ["image"]:
        img = cv2.imread(args.filePath)
        
        img = process_img(img, face_detection)
        cv2.imwrite(os.path.join('.', 'image_blured.jpg'), img)
    
    elif args.mode in ["video"]:
        cap =cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()
        output_video = cv2.VideoWriter(os.path.join('.', 'output.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 30, (frame.shape[1], frame.shape[0]))

        while True:
           
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

            if ret == False:
                break
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
    
    elif args.mode in ["webcam"]:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame = process_img(frame, face_detection)
            cv2.imshow("frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

#save image
#