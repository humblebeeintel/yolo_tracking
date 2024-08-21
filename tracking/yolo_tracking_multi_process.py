import os
import time
from multiprocessing import Pool, set_start_method
from track import run
from track import parse_opt
import warnings
import torch
from os.path import join, isdir, isfile
from itertools import cycle
import argparse
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

def check_images_in_dir(directory):
    supported_formats = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm')
    return any(isfile(join(directory, f)) and f.lower().endswith(supported_formats) for f in os.listdir(directory))

def process_sequence(seq_path, args, gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu')
    start_time = time.time()

    try:
        print(f'Processing sequence {seq_path} on GPU {gpu_id} as {device} (process ID: {os.getpid()})...', flush=True)
        
        args.source = seq_path
        args.device = device

        seq_name = os.path.basename(os.path.dirname(seq_path))
        args.name = seq_name

        if not os.path.exists(args.project):
            os.makedirs(args.project)

        run(args)
        
        end_time = time.time()
        print(f'Finished processing sequence {seq_path} on GPU {gpu_id} as {device} in {end_time - start_time:.2f} seconds', flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f'Error processing sequence {seq_path} on GPU {gpu_id} as {device} (process ID: {os.getpid()}): CUDA out of memory. {str(e)}', flush=True)
    except Exception as e:
        print(f'Error processing sequence {seq_path} on GPU {gpu_id} as {device} (process ID: {os.getpid()}): {str(e)}', flush=True)

def process_sequences_on_gpu(sequence_dirs, args, gpu_id):
    for seq_dir in tqdm(sequence_dirs, desc=f"GPU {gpu_id}"):
        process_sequence(seq_dir, args, gpu_id)

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    
    # Custom argument parser for multiprocessing control
    custom_parser = argparse.ArgumentParser(description="Run tracking on different datasets with various configurations")
    custom_parser.add_argument('--tracking-methods', type=str, nargs='+', required=True, help='List of tracking methods to use (deepocsort, botsort, ocsort, bytetrack)')
    custom_parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    custom_parser.add_argument('--split', type=str, choices=['test', 'train'], required=True, help='Dataset split to process (test or train)')
    custom_parser.add_argument('--gpu-ids', type=int, nargs='+', required=True, help='List of GPU IDs to use')
    custom_parser.add_argument('--workers-per-gpu', type=int, required=True, help='Number of workers per GPU')
    custom_parser.add_argument('--project', type=str, required=True, help='Directory to save the results')

    # Parse custom arguments first
    custom_args, remaining_args = custom_parser.parse_known_args()

    # Parse the remaining arguments using parse_opt
    opt_parser = argparse.ArgumentParser()
    opt_parser = parse_opt(opt_parser)
    args = opt_parser.parse_args(remaining_args)

    start_time = time.time()
    
    for tracking_method in custom_args.tracking_methods:
        args.tracking_method = tracking_method
        args.project = os.path.join(custom_args.project, f"{tracking_method}__input_{args.imgsz}__conf_{args.conf}")

        source_path = os.path.join(custom_args.dataset, custom_args.split)
        
        if os.path.isdir(source_path):
            sequence_dirs = [join(source_path, d, 'img1') for d in os.listdir(source_path) if isdir(join(source_path, d, 'img1'))]
            sequence_dirs = [d for d in sequence_dirs if check_images_in_dir(d)]
        else:
            raise ValueError(f"The provided source path {source_path} is not a directory")

        sequence_chunks = [[] for _ in range(len(custom_args.gpu_ids) * custom_args.workers_per_gpu)]

        for sequence, worker in zip(sequence_dirs, cycle(range(len(sequence_chunks)))):
            sequence_chunks[worker].append(sequence)

        for i, chunk in enumerate(sequence_chunks):
            print(f"Chunk {i+1} assigned to worker {i % custom_args.workers_per_gpu} on GPU {custom_args.gpu_ids[i // custom_args.workers_per_gpu]}: {chunk}")

        with Pool(processes=len(sequence_chunks)) as pool:
            results = []
            for i, chunk in enumerate(sequence_chunks):
                gpu_id = custom_args.gpu_ids[i // custom_args.workers_per_gpu]
                print(f'Assigning worker {i % custom_args.workers_per_gpu} on GPU {gpu_id} to process chunk {i+1}/{len(sequence_chunks)} for tracking method {tracking_method}', flush=True)
                result = pool.apply_async(process_sequences_on_gpu, args=(chunk, args, gpu_id))
                results.append(result)

            for result in results:
                result.wait()

            pool.close()
            pool.join()

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time taken for the run: {total_time:.2f} seconds', flush=True)
