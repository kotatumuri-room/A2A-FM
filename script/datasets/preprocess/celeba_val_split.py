import os
import random


def move_random_files(source_dir, dest_dir, num_files=100):
    """
    Moves a specified number of random files from the source directory to the destination directory.

    Args:
        source_dir: The path to the source directory.
        dest_dir: The path to the destination directory.
        num_files: The number of files to move.
    """

    try:
        # Get a list of all files in the source directory
        files = os.listdir(source_dir)

        # Check if there are enough files to move
        if len(files) < num_files:
            raise ValueError(
                f"Not enough files in {source_dir}. Found {len(files)}, need {num_files}."
            )

        # Select random files
        random_files = random.sample(files, num_files)

        # Move the selected files
        for file in random_files:
            source_path = os.path.join(source_dir, file)
            dest_path = os.path.join(dest_dir, file)
            os.rename(source_path, dest_path)

        print(f"{num_files} files moved successfully from {source_dir} to {dest_dir}.")

    except FileNotFoundError as e:
        print(f"Error: Directory not found: {source_dir},{e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
source_directory = "./datasets/celebADialogEmbbeded/"
destination_directory = "./datasets/celebADialogEmbbeded_eval/"
number_of_files = 100

move_random_files(source_directory, destination_directory, number_of_files)