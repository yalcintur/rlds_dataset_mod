import os

def convert_py_to_txt(directory):
    """
    Convert all .py files in the given directory to a single .txt file.
    The .txt file will have the original filename as a header for each Python file.
    """
    all_content = ""
    txt_filename = f"{os.path.basename(directory)}.txt"  # Use the directory name for the output file

    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            py_filepath = os.path.join(directory, filename)

            with open(py_filepath, "r", encoding="utf-8") as py_file:
                all_content += f"# {filename}\n\n"  # Header with filename
                all_content += py_file.read() + "\n\n"  # Add content and a newline separator

    # Define the path for the output .txt file
    txt_filepath = os.path.join(directory, txt_filename)

    # Write all collected content to the single .txt file
    with open(txt_filepath, "w", encoding="utf-8") as txt_file:
        txt_file.write(all_content)

    print(f"Converted all .py files into {txt_filename}")

# Set the directory containing .py files
directory = "/Users/yalcintur/LabWorkspace/rlds_dataset_mod"  # Change this to your directory path if needed

convert_py_to_txt(directory)
