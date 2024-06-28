from src.video_processor import VideoProcessor
from src.model_inference import ModelInference
from src.dataset_generator import DatasetGenerator
from src.dataset_validator import DatasetValidator

def main():
    video_path = "dataset/input/video.avi"
    output_frames_folder = "dataset/output/frames"
    output_labels_folder = "dataset/output/labels"

    classes = ["orange/red hollow circle", "orange/red hollow triangle", "orange/red hollow square", 
               "yellow hollow circle", "yellow hollow triangle", "yellow hollow square",
               'light green balloon', 'light purple balloon', 'blue blimp', 'red blimp']  

    VideoProcessor.split_video(video_path, output_frames_folder)

    model_inference = ModelInference()
    dataset_generator = DatasetGenerator(model_inference)
    dataset_generator.process_dataset(output_frames_folder, classes, output_labels_folder)

    # Validate dataset
    DatasetValidator.validate_dataset(output_frames_folder, output_labels_folder)

    print("Dataset generation and validation complete.")

if __name__ == "__main__":
    main()
