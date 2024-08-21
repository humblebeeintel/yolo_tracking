import os
import time
import torch
from multiprocessing import Pool, set_start_method
from track import run, parse_opt
import warnings
from os.path import join, isdir, isfile
from itertools import cycle

warnings.filterwarnings("ignore")

def check_images_in_dir(directory):
    supported_formats = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm')
    return any(isfile(join(directory, f)) and f.lower().endswith(supported_formats) for f in os.listdir(directory))

def process_sequence(seq_path, args, gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    start_time = time.time()

    try:
        print(f'Processing sequence {seq_path} on GPU {gpu_id} as {device} (process ID: {os.getpid()})...', flush=True)
        
        args.source = seq_path
        args.device = device
        args.yolo_model = "yolov8m"
        args.classes = 0
        args.save_txt = True
        args.tracking_method = 'bytetrack'
        args.project = os.path.join('/media/hbai/data/code/LiteSORT/yolo_tracking/hbai_scripts/PersonPath22_new-run', f"{args.tracking_method}__input_1280__conf_.25")
        args.name = os.path.basename(seq_path.split('/')[-2])
        args.exist_ok = True
        # print("args", args)
        run(args)
        
        end_time = time.time()
        print(f'Finished processing sequence {seq_path} on GPU {gpu_id} as {device} in {end_time - start_time:.2f} seconds', flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f'Error processing sequence {seq_path} on GPU {gpu_id} as {device} (process ID: {os.getpid()}): CUDA out of memory. {str(e)}', flush=True)
    except Exception as e:
        print(f'Error processing sequence {seq_path} on GPU {gpu_id} as {device} (process ID: {os.getpid()}): {str(e)}', flush=True)

def process_sequences_on_gpu(sequence_dirs, args, gpu_id):
    for seq_dir in sequence_dirs:
        process_sequence(seq_dir, args, gpu_id)

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    start_time = time.time()
    args = parse_opt()
    gpu_ids = [0]  # List of GPU indices to use
    workers_per_gpu = 12

    source_path = '/media/hbai/data/code/LiteSORT/datasets/PersonPath22/test'

    if os.path.isdir(source_path):
        sequence_dirs = [join(source_path, d, 'img1') for d in os.listdir(source_path) if isdir(join(source_path, d, 'img1'))]
        sequence_dirs = [d for d in sequence_dirs if check_images_in_dir(d)]
    else:
        raise ValueError("The provided source path is not a directory")

    sequence_chunks = [[] for _ in range(len(gpu_ids) * workers_per_gpu)]

    for sequence, worker in zip(sequence_dirs, cycle(range(len(sequence_chunks)))):
        sequence_chunks[worker].append(sequence)

    for i, chunk in enumerate(sequence_chunks):
        gpu_id = gpu_ids[i // workers_per_gpu]
        print(f'Assigning worker {i % workers_per_gpu} on GPU {gpu_id} to process chunk {i+1}/{len(sequence_chunks)}', flush=True)
        process_sequences_on_gpu(chunk, args, gpu_id)

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time taken for the run: {total_time:.2f} seconds', flush=True)
