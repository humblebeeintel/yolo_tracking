import os
import subprocess
from tqdm import tqdm

trackers = ['bytetrack', 'botsort', 'ocsort', 'deepocsort']

def run_tracking(tracker, video_name, video_path, output_dir):
    yolo_model = "yolov8m"  # Adjust if needed
    tracker_name = tracker.replace('_', '-')
    tracker_output_dir = os.path.join(output_dir, f"{tracker_name}__input_1280__conf_.25")
    
    if not os.path.exists(tracker_output_dir):
        os.makedirs(tracker_output_dir)
    
    tracking_command = (
        f"python /workspace/LiteSORT/yolo_tracking/tracking/track.py --tracking-method {tracker} --yolo-model {yolo_model} "
        f"--source {video_path} --classes 0 --save-txt --project {tracker_output_dir} "
        f"--name {video_name} --exist-ok --device '0' --verbose"
    )
    #print(tracking_command)
    
    # Run the tracking command and capture the output in real-time
    process = subprocess.Popen(tracking_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    frame_count = 0
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            frame_count += 1
            if frame_count % 100 == 0:
                print(output.strip())  # Print the output to the terminal every 100 frames

    # Save the output log for reference
    stdout, stderr = process.communicate()
    with open(os.path.join(tracker_output_dir, 'tracking_output.log'), 'w') as log_file:
        log_file.write(stdout if stdout else '')
        log_file.write(stderr if stderr else '')

def main(dataset_dir, output_dir):
    train_dir = os.path.join(dataset_dir, 'train')
    video_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for tracker in trackers:
        for video_dir in tqdm(video_dirs, desc=f"Processing videos with {tracker}", unit="video"):
            video_path = os.path.join(train_dir, video_dir, f"{video_dir}.mp4")
            if os.path.exists(video_path):
                run_tracking(tracker, video_dir, video_path, output_dir)
            else:
                print(f"Video file not found: {video_path}")

# Specify the dataset directory and output directory
dataset_dir = '/workspace/LiteSORT/datasets/MOT20'
output_dir = '/workspace/LiteSORT/yolo_tracking/hbai_scripts/MOT20'

# Run the tracking for all videos in the dataset with all trackers
main(dataset_dir, output_dir)
