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
    start_time = time.time()

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

    end_time = time.time()
    print(
        f'Finished processing sequence {seq_path} on {device} in {end_time - start_time:.2f} seconds', flush=True)
    # calculate FPS
    print(
        f'len(os.listdir(os.path.join(args.project, seq_name))): {len(os.listdir(os.path.join(args.project, seq_name))):.2f}', flush=True)
    print(f'FPS: {len(os.listdir(os.path.join(args.project, seq_name))) / (end_time - start_time):.2f}', flush=True)


if __name__ == '__main__':
    start_time = time.time()

    args = parse_opt()

    args.dataset = os.path.join(os.getcwd(), 'datasets', args.dataset)
    print(f'args.dataset: {args.dataset}')
    # Ensure the source is a directory containing subdirectories with image sequences
    if os.path.isdir(args.dataset):

        path_to_sequences = join(args.dataset, args.split)

        sequence_dirs = [join(path_to_sequences, d, 'img1') for d in os.listdir(
            path_to_sequences) if isdir(join(path_to_sequences, d, 'img1'))]
        # Filter out directories without supported images
        sequence_dirs = [d for d in sequence_dirs if check_images_in_dir(d)]
    else:
        raise ValueError("The provided source path is not a directory")

    print(
        f"Processing with Conf threshold {args.conf} using tracker {args.tracking_method}...")

    for i, seq_dir in enumerate(sequence_dirs):

        process_sequence(seq_dir, args)

    end_time = time.time()
    total_time = end_time - start_time
    print(
        f'Total time taken for the run: {total_time:.2f} seconds', flush=True)
