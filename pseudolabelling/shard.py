import os
import tarfile
import argparse
from pathlib import Path
import shutil

def extract_tar(input_tar, temp_folder):
    """Extracts a tar file into a temporary folder."""
    os.makedirs(temp_folder, exist_ok=True)
    
    with tarfile.open(input_tar, "r") as tar:
        tar.extractall(temp_folder)

    print(f"Extracted {input_tar} to {temp_folder}")

def create_sharded_tars(temp_folder, output_folder, num_shards):
    """Splits extracted files into multiple tar shards."""
    os.makedirs(output_folder, exist_ok=True)

    # Get all extracted files
    files = sorted(Path(temp_folder).rglob("*"))  # Recursively get all files
    files = [f for f in files if f.is_file()]  # Exclude directories
    total_files = len(files)

    if total_files == 0:
        print("No files found in extracted dataset!")
        return

    # Compute shard size
    files_per_shard = (total_files + num_shards - 1) // num_shards  # Round up

    print(f"Total extracted files: {total_files}")
    print(f"Files per shard: {files_per_shard}")

    # Create shards
    for shard_idx in range(num_shards):
        start_idx = shard_idx * files_per_shard
        end_idx = min(start_idx + files_per_shard, total_files)

        if start_idx >= total_files:
            break  # Avoid empty shards

        shard_filename = os.path.join(output_folder, f"{shard_idx:05d}.tar")

        with tarfile.open(shard_filename, "w") as tar:
            for file_path in files[start_idx:end_idx]:
                tar.add(file_path, arcname=file_path.relative_to(temp_folder))

        print(f"Shard {shard_idx} created: {shard_filename} ({end_idx - start_idx} files)")

def main():
    parser = argparse.ArgumentParser(description="Shard a large .tar file into smaller .tar files.")
    parser.add_argument("--input-tar", type=str, required=True, help="Path to the input .tar file.")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to store the sharded .tar files.")
    parser.add_argument("--num-shards", type=int, required=True, help="Number of shards to create.")
    parser.add_argument("--temp-folder", type=str, default="./temp_extract", help="Temporary extraction folder.")

    args = parser.parse_args()

    # Step 1: Extract the tar file to a temporary directory
    extract_tar(args.input_tar, args.temp_folder)

    # Step 2: Create sharded tar files
    create_sharded_tars(args.temp_folder, args.output_folder, args.num_shards)

    shutil.rmtree(args.temp_folder)

if __name__ == "__main__":
    main()
