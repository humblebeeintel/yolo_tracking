import os
import subprocess
from tqdm import tqdm

trackers = ['deepocsort', 'botsort']
confidence_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def run_tracking(tracker, video_name, video_path, output_dir, conf_threshold):
    yolo_model = "yolov8x"  # Adjust if needed
    tracker_name = tracker.replace('_', '-')
    tracker_output_dir = os.path.join(output_dir, f"lite_{tracker_name}__input_1280__conf_{conf_threshold}")
    
    if not os.path.exists(tracker_output_dir):
        os.makedirs(tracker_output_dir)
    
    tracking_command = (
        f"python3 /media/hbai/data/code/LiteSORT/yolo_tracking/tracking/track.py --tracking-method {tracker} --yolo-model {yolo_model} "
        f"--source {video_path} --classes 0 --save-txt --project {tracker_output_dir} "
        f"--name {video_name} --exist-ok --verbose --appearance-feature-layer layer0 --conf {conf_threshold}"
    )
    print(tracking_command)
    
    # Run the tracking command and capture the output in real-time
    process = subprocess.Popen(tracking_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    frame_count = 0
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            frame_count += 1
            print(output.strip())  # Print the output to the terminal

def main(dataset_dir, output_dir):
    test_dir = os.path.join(dataset_dir, 'test')
    seq_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    for tracker in trackers:
        for conf_threshold in confidence_thresholds:
            for video_dir in tqdm(seq_dirs, desc=f"Processing sequence with {tracker} at conf {conf_threshold}", unit="video"):
                video_path = os.path.join(test_dir, video_dir, "img1")
                if os.path.exists(video_path):
                    print(f'Processing video: {video_path}')
                    run_tracking(tracker, video_dir, video_path, output_dir, conf_threshold)
                else:
                    print(f"Video file not found: {video_path}")

# Specify the dataset directory and output directory
dataset_dir = '/media/hbai/data/code/LiteSORT/datasets/PersonPath22'
output_dir = '/media/hbai/data/code/LiteSORT/results/conf_thresh_experiment/PersonPath22-test_LITE'

# Run the tracking for all videos in the dataset with all trackers and confidence thresholds
main(dataset_dir, output_dir)
