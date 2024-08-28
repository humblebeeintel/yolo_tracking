import os
import time
from multiprocessing import Pool, set_start_method
from track import run
from track import parse_opt
import torch
from os.path import join, isdir, isfile
from itertools import cycle


def check_images_in_dir(directory):
    supported_formats = ('.jpg', '.png')
    return any(isfile(join(directory, f)) and f.lower().endswith(supported_formats) for f in os.listdir(directory))


def process_sequence(seq_path, args, gpu_id):
    torch.cuda.set_device(gpu_id)

    device = torch.device(f'cuda:{gpu_id}')
    start_time = time.time()

    try:
        print(
            f'Processing sequence {seq_path} on GPU {gpu_id} as {device} (process ID: {os.getpid()})...', flush=True)

        args.source = seq_path
        args.device = device
        args.yolo_model = "yolov8x"
        args.classes = 0
        args.save_txt = True
        args.exist_ok = True
        args.appearance_feature_layer = "layer0"

        seq_name = os.path.basename(os.path.dirname(seq_path))
        args.name = seq_name

        if not os.path.exists(args.project):
            print(f'Creating project directory {args.project}', flush=True)
            os.makedirs(args.project)

        run(args)

        end_time = time.time()
        print(
            f'Finished processing sequence {seq_path} on GPU {gpu_id} as {device} in {end_time - start_time:.2f} seconds', flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(
            f'Error processing sequence {seq_path} on GPU {gpu_id} as {device} (process ID: {os.getpid()}): CUDA out of memory. {str(e)}', flush=True)
    except Exception as e:
        print(
            f'Error processing sequence {seq_path} on GPU {gpu_id} as {device} (process ID: {os.getpid()}): {str(e)}', flush=True)


def process_sequences_on_gpu(sequence_dirs, args, gpu_id):
    for seq_dir in sequence_dirs:
        process_sequence(seq_dir, args, gpu_id)


if __name__ == '__main__':
    set_start_method('spawn', force=True)

    start_time = time.time()

    args = parse_opt()

    gpu_ids = [0]
    workers_per_gpu = 5

    conf_thresh = 0.25

    source_path = '/home/humblebee/code/LITE/datasets/MOT17/train'

    # Ensure the source is a directory containing subdirectories with image sequences
    if os.path.isdir(source_path):
        sequence_dirs = [join(source_path, d, 'img1') for d in os.listdir(
            source_path) if isdir(join(source_path, d, 'img1'))]
        # Filter out directories without supported images
        sequence_dirs = [d for d in sequence_dirs if check_images_in_dir(d)]
    else:
        raise ValueError("The provided source path is not a directory")

    trackers = ['botsort', 'deepocsort']

    for tracker in trackers:
        args.tracking_method = tracker
        args.project = os.path.join(
            '/home/humblebee/code/LITE/results/MOT20', f"lite_{tracker}__input_1280__conf_{conf_thresh}")

        print(
            f"Processing with Conf threshold {conf_thresh} using tracker {tracker}...")
        args.conf = conf_thresh

        sequence_chunks = [[]
                           for _ in range(len(gpu_ids) * workers_per_gpu)]

        for sequence, worker in zip(sequence_dirs, cycle(range(len(sequence_chunks)))):
            sequence_chunks[worker].append(sequence)

        for i, chunk in enumerate(sequence_chunks):
            print(
                f"Chunk {i+1} assigned to worker {i % workers_per_gpu} on GPU {gpu_ids[i // workers_per_gpu]}: {chunk}")

        with Pool(processes=len(sequence_chunks)) as pool:
            results = []
            for i, chunk in enumerate(sequence_chunks):
                gpu_id = gpu_ids[i // workers_per_gpu]
                print(
                    f'Assigning worker {i % workers_per_gpu} on GPU {gpu_id} to process chunk {i+1}/{len(sequence_chunks)} for tracker {tracker}', flush=True)
                result = pool.apply_async(
                    process_sequences_on_gpu, args=(chunk, args, gpu_id))
                results.append(result)

            for result in results:
                result.wait()

            pool.close()
            pool.join()

    end_time = time.time()
    total_time = end_time - start_time
    print(
        f'Total time taken for the run: {total_time:.2f} seconds', flush=True)
