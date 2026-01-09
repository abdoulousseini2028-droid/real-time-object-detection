import cv2
import torch
import yt_dlp
import numpy as np

class ObjectDetection:
    def __init__(self, youtube_url, out_file="github_demo.mp4"):
        # 1. Define URL and Load Model
        self.url = youtube_url
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()
        self.classes = self.model.names
        print(f"Using device: {self.device}")

    def load_model(self):
        # 2026 Update: force_reload=False to use cache, .to(device) for speed
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(self.device)
        return model

    def get_video_stream(self):
        # yt-dlp is the most reliable extractor for YouTube in 2026
        ydl_opts = {'format': 'best[height<=480]', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.url, download=False)
            return cv2.VideoCapture(info['url'])

    def score_frame(self, frame):
        # Perform detection
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.25: # Confidence Threshold
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw Label
                label_text = f"{self.classes[int(labels[i])]} {row[4]:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def __call__(self):
        cap = self.get_video_stream()
        
        # --- SAFE VIDEO WRITER INITIALIZATION ---
        # Read the first frame to get EXACT pixel dimensions
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not access video stream.")
            return

        height, width, _ = frame.shape
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # 'mp4v' or 'avc1' are best for macOS 2026 compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.out_file, fourcc, fps, (width, height))

        print(f"Recording to {self.out_file}. PRESS 'q' TO STOP AND SAVE.")

        while cap.isOpened():
            # Process the frame we just read
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            # Write frame to file
            out.write(frame)

            # Show the frame
            cv2.imshow('YOLOv5 YouTube Detection', frame)

            # Read next frame
            ret, frame = cap.read()
            if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        
        # --- CLEANUP ---
        cap.release()
        out.release() # CRITICAL: This saves the file to disk
        cv2.destroyAllWindows()
        print(f"\nSuccessfully saved: {self.out_file}")

# --- RUN THE APP ---
if __name__ == "__main__":
    # You can change this link to any YouTube video
    youtube_link = "https://www.youtube.com/watch?v=ddPnEk90vLk" 
    
    # Initialize and run
    detector_instance = ObjectDetection(youtube_link, out_file="github_demo.mp4")
    detector_instance()


