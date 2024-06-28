import os
from .model_inference import ModelInference

class DatasetGenerator:
    def __init__(self, model_inference):
        self.model_inference = model_inference

    def process_dataset(self, image_folder, classes, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for image_file in os.listdir(image_folder):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_folder, image_file)
                results = self.model_inference.run_inference(image_path, classes)
                
                self._save_yolo_format(results, classes, image_file, output_folder)

    @staticmethod
    def _save_yolo_format(results, classes, image_file, output_folder):
        with open(f"{output_folder}/{image_file.split('.')[0]}.txt", 'w') as f:
            for bbox, label in zip(results['bboxes'], results['labels']):
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                class_id = classes.index(label)
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
