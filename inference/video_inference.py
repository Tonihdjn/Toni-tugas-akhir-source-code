import cv2
from ultralytics import YOLO

def inference_video(model_path, video_path, output_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame)
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)
        cv2.imshow('Armed Person Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference_video('hasil/Baseline/yolov10/detect/train/weights/best.pt', 
                   'video_for_inference.mp4',
                   'Best_yolov10.mp4')