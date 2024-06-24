import os
import time
from multiprocessing import Pool, set_start_method
from track import run
from track import parse_opt
import warnings
import torch
from os.path import join, isdir, isfile

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

        args.tracking_method = 'deepocsort'
        args.project = os.path.join('/workspace/LiteSORT/yolo_tracking/hbai_scripts/PersonPath22', f"{args.tracking_method}__input_1280__conf_.25")
        video_name = seq_path.split('_')[2]
        print(f'VIDEO_NAME: {video_name}')
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
        print(f'SEQ_DIR: {seq_dir}')

        process_sequence(seq_dir, args, gpu_id)

if __name__ == '__main__':
    # Set the multiprocessing start method to 'spawn'
    set_start_method('spawn', force=True)
    
    start_time = time.time()

    args = parse_opt()

    gpu_ids = [0, 1, 2, 3]  # List of GPU indices to use
    source_path = '/workspace/LiteSORT/datasets/PersonPath22/test'

    # Ensure the source is a directory containing subdirectories with image sequences
    if os.path.isdir(source_path):
        sequence_dirs = [join(source_path, d, 'img1') for d in os.listdir(source_path) if isdir(join(source_path, d, 'img1'))]
        # Filter out directories without supported images
        sequence_dirs = [d for d in sequence_dirs if check_images_in_dir(d)]
    else:
        raise ValueError("The provided source path is not a directory")

    # Debug print to check the collected directories
    # print(f"Collected sequence directories: {sequence_dirs}")

    # Limit to 8 sequences only
    sequence_dirs = sequence_dirs[:8]
    print(f"Selected sequence directories: {sequence_dirs}")

    # Split sequences into chunks, one for each GPU
    chunk_size = len(sequence_dirs) // len(gpu_ids)
    sequence_chunks = [
        sequence_dirs[i * chunk_size: (i + 1) * chunk_size] for i in range(len(gpu_ids))]

    # Ensure all sequences are included in case of uneven division
    if len(sequence_dirs) % len(gpu_ids) != 0:
        sequence_chunks[-1].extend(sequence_dirs[len(gpu_ids) * chunk_size:])

    # Debug print to check the sequence chunks
    for i, chunk in enumerate(sequence_chunks):
        print(f"Chunk {i+1} assigned to GPU {gpu_ids[i]}: {chunk}")

    # Use multiprocessing Pool with the same number of processes as GPUs
    with Pool(processes=len(gpu_ids)) as pool:
        results = []
        for i, chunk in enumerate(sequence_chunks):
            gpu_id = gpu_ids[i]
            print(f'Assigning GPU {gpu_id} to process chunk {i+1}/{len(sequence_chunks)}', flush=True)
            result = pool.apply_async(process_sequences_on_gpu, args=(chunk, args, gpu_id))
            results.append(result)

        for result in results:
            result.wait()

        pool.close()
        pool.join()

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time taken for the run: {total_time:.2f} seconds', flush=True)
