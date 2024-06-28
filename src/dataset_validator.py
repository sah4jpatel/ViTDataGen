import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DatasetValidator:
    @staticmethod
    def validate_dataset(image_folder, label_folder):
        for image_file in os.listdir(image_folder):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_folder, image_file)
                label_path = os.path.join(label_folder, f"{image_file.split('.')[0]}.txt")
                
                DatasetValidator._validate_image(image_path, label_path)

    @staticmethod
    def _validate_image(image_path, label_path):
        image = Image.open(image_path)
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = x_center - width/2
            y1 = y_center - height/2
            rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        plt.show()
        
        key = input("Press SPACE for correct, X for incorrect: ")
        if key.lower() == 'x':
            # Remove the incorrect bounding box
            lines.pop()
            with open(label_path, 'w') as f:
                f.writelines(lines)
            print(f"Removed last bounding box from {label_path}")
        
        plt.close()
