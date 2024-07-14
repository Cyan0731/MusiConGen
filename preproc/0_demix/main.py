'''
two-stems separation
'''
import os
from pathlib import Path
import uuid
import subprocess

import time
import shutil
import datetime


def traverse_dir(
        root_dir,
        extension,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


if __name__ == '__main__':
    path_rootdir = '../audiocraft/dataset/example/full'

    st_idx, ed_idx = 0, None
    ext_src = 'mp3'
    ext_dst = 'wav'

    # list files
    filelist = traverse_dir(
        path_rootdir,
        extension='mp3',
        str_include='',
        is_sort=True)
    num_file = len(filelist)
    if ed_idx is None:
        ed_idx = num_file
    print(' [i] num files:', num_file)

    # make tmpdir for demucs
    tmp_dir = os.path.join('tmp', str(uuid.uuid4()).split('-')[0])
    print('tmp_dir:', tmp_dir)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    # start running
    for i in range(st_idx, ed_idx):
        print(f'==={i}/{num_file}  [{st_idx} - {ed_idx}]====================')
        start_time = time.time()
        
        # path
        srcfile = filelist[i]
        print(srcfile)
        srcfile_dir = os.path.dirname(srcfile)
        source_folder = os.path.join(tmp_dir, 'htdemucs', 'full')
        path_src_vocals = os.path.join(source_folder, f'vocals.{ext_dst}')
        path_src_no_vocals = os.path.join(source_folder, f'no_vocals.{ext_dst}')
        path_dst_vocals = os.path.join(srcfile_dir, f'vocals.{ext_dst}')
        path_dst_no_vocals =  os.path.join(srcfile_dir, f'no_vocal.{ext_dst}')

        if os.path.exists(path_dst_no_vocals):
            print('[o] existed')
            continue

        # source separation
        cmd_list = [
            'demucs', 
            '--two-stems=vocals', 
            f'{srcfile}',
            '-o',  
            f'{tmp_dir}'
          ]
        
        if ext_dst == 'mp3':
            print('[i] save in mp3 format')
            cmd_list.append('--mp3')
        subprocess.run(cmd_list)
        
        # copy from tmp to dst
        shutil.copy2(path_src_vocals, path_dst_vocals)
        shutil.copy2(path_src_no_vocals, path_dst_no_vocals)

        # end
        end_time = time.time()
        runtime = end_time - start_time
        print('testing time:', str(datetime.timedelta(seconds=runtime))+'\n')

    shutil.rmtree(tmp_dir)