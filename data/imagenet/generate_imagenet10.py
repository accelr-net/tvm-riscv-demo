import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Process images in subdirectories.')
parser.add_argument('source_dir', type=str, help='Path to the source directory')
args = parser.parse_args()

# Hardcoded destination directory
destination_dir = './data/imagenet/imagenet10/val'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Get subdirectories
subdirs = [d for d in os.listdir(args.source_dir) if os.path.isdir(os.path.join(args.source_dir, d))]

for subdir in subdirs:
  src_subdir = os.path.join(args.source_dir, subdir)
  dest_subdir = os.path.join(destination_dir, subdir)

  # Create directories in the destination location
  os.makedirs(dest_subdir, exist_ok=True)

  # Get file names in the source directory
  image_list = [f for f in os.listdir(src_subdir) if os.path.isfile(os.path.join(src_subdir, f))]

  # Extract and sort numbers from image names
  sorted_numbers = sorted([int(f.split('_')[-1].split('.')[0]) for f in image_list])

  # Select the lowest five numbers
  moveable_numbers = sorted_numbers[:5]

  # Create a list of movable images
  moveable_images = [f"ILSVRC2012_val_{str(num).zfill(8)}.JPEG" for num in moveable_numbers]

  # Copy the selected images to the destination directory
  for image in moveable_images:
    src_path = os.path.join(src_subdir, image)
    dest_path = os.path.join(dest_subdir, image)
    shutil.copy(src_path, dest_path)
