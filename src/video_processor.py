import cv2
import os

class VideoProcessor:
    @staticmethod
    def split_video(video_path, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(f"{output_folder}/frame{count:05d}.jpg", image)
            success, image = vidcap.read()
            count += 1
        print(f"Extracted {count} frames")
        return count
