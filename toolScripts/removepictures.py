import os

def remove_non_jpeg_files(directory):
    """
    Removes all files in the specified directory that are not JPEG or JPG.

    Args:
        directory (str): Path to the directory to scan.
    """
    try:
        # List all files in the directory
        files = os.listdir(directory)

        for file in files:
            # Construct full file path
            file_path = os.path.join(directory, file)

            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                # Check file extension
                if not file.lower().endswith(('.jpeg', '.jpg')):
                    print(f"Removing: {file}")
                    os.remove(file_path)  # Remove the file

        print("Operation completed. Non-JPEG files have been removed.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
directory_path = "data\dataset"  # Replace with the path to your directory
remove_non_jpeg_files(directory_path)