import pandas as pd
import multiprocessing as mp
import os
from news_process.tools.constants import PATH_DATA

NUM_CORES = 2

def main_multi_threading_process(sections_chunk: list, 
                                 pid: str
                                 ):
    pass

def _split_files(file_path: str, 
                 n_splits: int
                 ):
    with open(file_path, "r") as f:
        data = f.readlines()
    # Split the input file into chunks
    chunk_size = len(data) // n_splits
    end = len(data)
    remainder = len(data) % n_splits
    if remainder > 0:
        end -= remainder
    chunks = []
    for i in range(0, end, chunk_size):
        if (i + chunk_size) < end:
            chunk = data[i:i + chunk_size]
        else:
            chunk = data[i:i + chunk_size + remainder]
        chunks.append(chunk)
    return chunks

def split_file_and_process(sections_file_path: str, 
                           num_splits: int, 
                           process_function):
    
    chunks = _split_files(sections_file_path, 
                          num_splits)

    # Create a process for each chunk and run in parallel
    processes = []
    for i, chunk in enumerate(chunks):
        split_file_path = os.path.join(PATH_DATA, f"sections_split_{i}.txt")  
        with open(split_file_path, 'w') as split_file:
            split_file.write("\n".join(chunk))

        process = mp.Process(target=process_function, 
                             args=(chunk, 
                                   str(i),
                                   )
                                   )
        processes.append(process)
        process.start()
    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Clean up the split files (optional)
    for i in range(num_splits):
        split_file_path = os.path.join(PATH_DATA, f"sections_split_{i}.txt")
        os.remove(split_file_path)

if __name__ == "__main__":
    file_path = os.path.join(PATH_DATA, f"test_file")

    split_file_and_process(file_path, 
                           NUM_CORES, 
                           main_multi_threading_process)
    print("\n...The process ended")
    

