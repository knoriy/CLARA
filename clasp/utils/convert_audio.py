import os
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SRC_FILE_EXT = '.webm'
DEST_FILE_EXT = '.flac'

def audio_to_flac(audio_in_path, audio_out_path, sample_rate=48000, no_log=True, segment_start:float=0, segment_end:float=None):
    log_cmd = ' -v quiet' if no_log else ''
    segment_cmd = f'-ss {segment_start} -to {segment_end}' if segment_end else ''
    os.system(
        f'ffmpeg -y -i "{audio_in_path}" -vn {log_cmd} -flags +bitexact '
        f'-ar {sample_rate} -ac 1 {segment_cmd} "{audio_out_path}"')

def create_threads(src_paths, dest_root_path, max_workers=96):
    if not os.path.exists(dest_root_path):
        raise FileNotFoundError(f'Please Check {dest_root_path} exists')

    l = len(src_paths)
    with tqdm.tqdm(total=l, desc=f'Processing {dest_root_path}') as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            threads = []
            for path in src_paths:
                folder_name = os.path.basename(os.path.dirname(path))
                file_name = os.path.basename(path)
                folder_and_file_name = f"{folder_name}/{file_name}"
                dest_path = os.path.join(dest_root_path, folder_and_file_name.replace(SRC_FILE_EXT, DEST_FILE_EXT))
                print(dest_path)
                threads.append(executor.submit(audio_to_flac, path, dest_path))

            for _ in as_completed(threads):
                pbar.update(1)


if __name__ == '__main__':
    import glob
    src_paths = glob.glob('/fsx/knoriy/raw_datasets/EMNS/cleaned_webm/**/*.webm', recursive=True)
    dest_root_path = '/fsx/knoriy/raw_datasets/EMNS/'

    create_threads(src_paths, dest_root_path, max_workers=96)