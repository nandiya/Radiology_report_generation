import os

def list_jpg_files(directory, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg'):
                    file_path = os.path.join(root, file)
                    f.write(file_path + '\n')

# Example usage:
directory_to_search = '/group/pmc023/rnandiya/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
output_txt_file = 'jpg_files_list.txt'

list_jpg_files(directory_to_search, output_txt_file)

print(f"List of .jpg files written to {output_txt_file}")
