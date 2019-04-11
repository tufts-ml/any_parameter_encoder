import glob
import os
import shutil


abs_path = os.getcwd()
dir = os.path.join(abs_path, 'vae_run3')

for filename in os.listdir(dir):
    # file is a saves h5 file
    if filename.endswith('.h5'):
        if '_hallucinations' in filename:
            name_parts = filename.split('_hallucinations')
            new_name = ''.join([name_parts[0], '_hallucinations', name_parts[-1]])
            os.rename(os.path.join(dir, filename), os.path.join(dir, new_name))
    else:
        # "file" is a directory
        if filename.count('_hallucinations') > 1:
            destination = filename.split('_hallucinations')[0] + '_hallucinations'
            for file in os.listdir(os.path.join(dir, filename)):
                shutil.move(os.path.join(dir, filename), os.path.join(dir, destination))

