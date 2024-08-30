import os
import time
import torch
from os.path import join, isdir, isfile

from track import run
from track import parse_opt


def check_images_in_dir(directory):
    supported_formats = ('.jpg', '.png')
    return any(isfile(join(directory, f)) and f.lower().endswith(supported_formats) for f in os.listdir(directory))


def process_sequence(seq_path, args, gpu_id=0):
    torch.cuda.set_device(gpu_id)

    device = torch.device(f'cuda:{gpu_id}')
    tick = time.time()

    print(f'Processing sequence {seq_path} on {device}', flush=True)
    
    args.source = seq_path
    args.device = device
    args.yolo_model = "yolov8m"
    args.classes = 0
    args.save_txt = True
    args.exist_ok = True

    seq_name = os.path.basename(os.path.dirname(seq_path))
    args.name = seq_name

    if not os.path.exists(args.project):
        print(f'Creating project directory {args.project}', flush=True)
        os.makedirs(args.project)


    run(args)


    tock = time.time()

    num_frames = len(os.listdir(join(seq_path)))
    print(f'Number of frames: {num_frames}')

    time_spent_for_the_sequence = tock - tick

    avg_time_per_frame = (time_spent_for_the_sequence) / num_frames
    
    FPS = 1/avg_time_per_frame
  
    path_to_fps_csv = f'results/{os.path.basename(args.dataset)}-FPS/{args.sequence}/fps.csv'
   
    if not os.path.exists(path_to_fps_csv):
        with open(path_to_fps_csv, 'w') as f:
            f.write('tracker_name,sequence_name,FPS\n')
            
    with open(path_to_fps_csv, 'a') as f:
        if args.appearance_feature_layer is None:
            f.write(f'{args.tracking_method},{seq_name},{FPS:.1f}\n')
        else:
            f.write(f'{"LITE" + args.tracking_method},{seq_name},{FPS:.1f}\n')

    print(
        f'Finished processing sequence {seq_path} on {device} in {tock - tick:.2f} seconds', flush=True)


if __name__ == '__main__':

    args = parse_opt()

    args.dataset = os.path.join(os.getcwd(), 'datasets', args.dataset)

    # Ensure the source is a directory containing subdirectories with image sequences
    if os.path.isdir(args.dataset):
        seq_dir = os.path.join(args.dataset, args.split, args.sequence, 'img1') 
    else:
        raise ValueError("The provided source path is not a directory")

    process_sequence(seq_dir, args)
