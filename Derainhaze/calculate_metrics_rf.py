import os
import pyiqa
import sys

def calculate_average_metric(metric_name, folder_path):
    # Initialize the specified metric
    metric = pyiqa.create_metric(metric_name)
    
    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # List to store metric scores
    scores = []
    
    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            try:
                # Compute metric score
                score = metric(image_path)
                scores.append(score.item())
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    if scores:
        average_score = sum(scores) / len(scores)
        print(f"Average {metric_name.upper()}: {average_score}")
    else:
        print("No valid images found in the folder.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_metrics_rf.py <metric_name> <folder_path>")
        sys.exit(1)
    metric_name = sys.argv[1]
    folder_path = sys.argv[2]
    calculate_average_metric(metric_name, folder_path)