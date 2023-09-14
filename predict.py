import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'videos') #'videos' file is where test videos are saved in your computer -you can change it 

video_path = os.path.join(VIDEOS_DIR, 'vid.mp4') #'vid.mp4' is the name of the video that you are going to test the model with -you can change it
video_path_out = '{}_out.mp4'.format(video_path) #how and where the output video will be saved

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read() #read every frame
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold: #draw the bounding box if the score is bigger than the treshold value that you decided on line 22
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
