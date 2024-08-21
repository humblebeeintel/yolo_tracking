import os
import time
from multiprocessing import Pool, set_start_method
from track import run
from track import parse_opt
import warnings
import torch
from os.path import join, isdir, isfile
from itertools import cycle

warnings.filterwarnings("ignore")

def check_images_in_dir(directory):
    supported_formats = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm')
    return any(isfile(join(directory, f)) and f.lower().endswith(supported_formats) for f in os.listdir(directory))

def process_sequence(seq_path, args, gpu_id):
    # Explicitly set the current device to the specified GPU
    torch.cuda.set_device(gpu_id)

    device = torch.device(f'cuda:{gpu_id}')
    start_time = time.time()

    try:
        print(f'Processing sequence {seq_path} on GPU {gpu_id} as {device} (process ID: {os.getpid()})...', flush=True)
        
        # Update the source path to the current sequence path
        args.source = seq_path
        args.device = device
        args.yolo_model = "yolov8m"
        args.classes = 0
        args.save_txt = True

        # Tracking method
        args.tracking_method = 'deepocsort'
        args.project = os.path.join('/media/hbai/data/code/LiteSORT/yolo_tracking/hbai_scripts/MOT20-train_LITE', f"{args.tracking_method}__input_1280__conf_.25")
        video_name = seq_path.split('/')[-2]
        #print(f'VIDEO_NAME: {video_name}')
        args.name = os.path.basename(video_name)
        args.exist_ok = True

        run(args)
        
        end_time = time.time()
        print(f'Finished processing sequence {seq_path} on GPU {gpu_id} as {device} in {end_time - start_time:.2f} seconds', flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f'Error processing sequence {seq_path} on GPU {gpu_id} as {device} (process ID: {os.getpid()}): CUDA out of memory. {str(e)}', flush=True)
    except Exception as e:
        print(f'Error processing sequence {seq_path} on GPU {gpu_id} as {device} (process ID: {os.getpid()}): {str(e)}', flush=True)

def process_sequences_on_gpu(sequence_dirs, args, gpu_id):
    for seq_dir in sequence_dirs:
        #print(f'SEQ_DIR: {seq_dir}')
        process_sequence(seq_dir, args, gpu_id)

if __name__ == '__main__':
    # Set the multiprocessing start method to 'spawn'
    set_start_method('spawn', force=True)
    
    start_time = time.time()

    args = parse_opt()

    gpu_ids = [0]  # List of GPU indices to use
    workers_per_gpu = 4 # Number of workers per GPU

    source_path = '/media/hbai/data/code/LiteSORT/datasets/MOT20/train'

    # Ensure the source is a directory containing subdirectories with image sequences
    if os.path.isdir(source_path):
        sequence_dirs = [join(source_path, d, 'img1') for d in os.listdir(source_path) if isdir(join(source_path, d, 'img1'))]
        # Filter out directories without supported images
        sequence_dirs = [d for d in sequence_dirs if check_images_in_dir(d)]
    else:
        raise ValueError("The provided source path is not a directory")

    # Create an empty list for each worker
    sequence_chunks = [[] for _ in range(len(gpu_ids) * workers_per_gpu)]

    # Distribute sequences in a round-robin fashion
    for sequence, worker in zip(sequence_dirs, cycle(range(len(sequence_chunks)))):
        sequence_chunks[worker].append(sequence)

    # Debug print to check the sequence chunks
    for i, chunk in enumerate(sequence_chunks):
        print(f"Chunk {i+1} assigned to worker {i % workers_per_gpu} on GPU {gpu_ids[i // workers_per_gpu]}: {chunk}")

    # Use multiprocessing Pool with the total number of workers
    with Pool(processes=len(sequence_chunks)) as pool:
        results = []
        for i, chunk in enumerate(sequence_chunks):
            gpu_id = gpu_ids[i // workers_per_gpu]
            print(f'Assigning worker {i % workers_per_gpu} on GPU {gpu_id} to process chunk {i+1}/{len(sequence_chunks)}', flush=True)
            result = pool.apply_async(process_sequences_on_gpu, args=(chunk, args, gpu_id))
            results.append(result)

        for result in results:
            result.wait()

        pool.close()
        pool.join()

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time taken for the run: {total_time:.2f} seconds', flush=True)
