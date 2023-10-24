import os
import sys
sys.path.append("/fsx/knoriy/CLASP/clasp/")
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple, TypeVar
import yaml
import subprocess
import tempfile


import torchdata
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from torch.utils.data import DataLoader
import tqdm

from utils import get_s3_paths, get_lists
from datamodule.utils import Boto3FileOpenerIterDataPipe as Boto3FileOpener
from datamodule.utils import filepath_fn

from torchdata.datapipes.iter.util.cacheholder import CacheState, _get_list_filename, _hash_check, _promise_filename
from torchdata.datapipes.iter.util.cacheholder import OnDiskCacheHolderIterDataPipe
try:
    import portalocker
except ImportError:
    portalocker = None


class MOnDiskCacheHolderIterDataPipe(OnDiskCacheHolderIterDataPipe):
    @staticmethod
    def _cache_check_fn(data, filepath_fn, hash_dict, hash_type, extra_check_fn, cache_uuid):
        filepath = data if filepath_fn is None else filepath_fn(data)
        assert not isinstance(filepath, (list, tuple))  # BC breaking, now only str is accepted as return

        result = CacheState.CACHED_SINGLE_ENTITY
        cached_file_exists = True
        if os.path.exists(_get_list_filename(filepath)):
            return int(CacheState.CACHED_MULTIPLE_ENTITIES)
        if not os.path.exists(filepath):
            cached_file_exists = False
        elif hash_dict is not None and not _hash_check(filepath, hash_dict, hash_type):
            # TODO: It is safer to assume that entire cache is compromised and require user to wipe it
            cached_file_exists = False
        elif extra_check_fn is not None and not extra_check_fn(filepath):
            # TODO: It is safer to assume that entire cache is compromised and require user to wipe it
            cached_file_exists = False
        if not cached_file_exists:
            promise_filepath = _promise_filename(filepath, cache_uuid)
            dirname = os.path.dirname(promise_filepath)
            os.makedirs(dirname, exist_ok=True)

            with portalocker.Lock(promise_filepath, "a+", flags=portalocker.LockFlags.EXCLUSIVE) as promise_fh:
                promise_fh.seek(0)
                data = promise_fh.read()
                # TODO(VitalyFedyunin): Potentially there is old .promise file from previous failed run, we
                # need to somehow propagate uniq session id for dataloader, save and compare it here,
                # raising error
                file_exists = len(data) > 0
                if not file_exists:
                    result = CacheState.UNCACHED
                    promise_fh.seek(0)
                    data = promise_fh.read()
                    # TODO(635): Potentially there is old .promise file from previous failed run, we
                    # need to somehow propagate uniq session id for dataloader, save and compare it here,
                    # raising error
                    file_exists = len(data) > 0
                    if not file_exists:
                        promise_fh.seek(0)
                        promise_fh.write("[dataloader session uid]")
                        promise_fh.truncate()
                        promise_fh.flush()

        return int(result)

def download_s3_folder(root_s3_path, dataset_name, save_dir):
    command = f"aws s3 sync {os.path.join(root_s3_path, dataset_name, '')} {save_dir}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error downloading folder from S3: {stderr.decode()}")

def main(args):
    with open('/fsx/knoriy/CLASP/config/config/data/base.yaml') as file:
        config = yaml.full_load(file)['init_args']
    dataset_names = get_lists(config['dataset_list'])
    exclude_names = get_lists(config['exclude_list'])
    root_data_path = 's3://laion-west-audio/webdataset_tar/'

    if args.backend == 'awscli':
        temp_dir = tempfile.gettempdir()
        for dataset_name in tqdm.tqdm(dataset_names, desc='Downloading: '):
            save_dir = os.path.join(temp_dir, "CLASP", 'laion-west-audio/webdataset_tar/', dataset_name)
            download_s3_folder(root_s3_path=root_data_path, dataset_name=dataset_name, save_dir=save_dir)

    elif args.backend == 'tdm':
        root_data_path = root_data_path.replace('s3://', '')
        urls = get_s3_paths(
            base_path			= root_data_path,
            train_valid_test	= config['train_valid_test'],
            dataset_names		= dataset_names, 
            exclude				= exclude_names,
            recache				= True
            )
        
        updated_urls = []
        for key in urls.keys():
            updated_urls.extend(urls[key])
        
        datapipe = torchdata.datapipes.iter.IterableWrapper(updated_urls)\
            .sharding_filter()\

        datapipe = MOnDiskCacheHolderIterDataPipe(datapipe, filepath_fn=filepath_fn)\
            .open_files_by_boto3(mode='rb')\
            .end_caching(filepath_fn=filepath_fn)\
            
        dl = DataLoader(datapipe)
        
        for _ in tqdm.tqdm(dl, total=len(updated_urls), desc='Downloading'):
            pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, help='Path to data config file')
    parser.add_argument('--backend', type=str, choices=['tdm', 'awscli'], help='what backend to use, aws cli with download files using the cli or tdm will use torchdata to download files')

    args = parser.parse_args()
    main(args)

    # python scripts/download_from_s3.py --data ./config/config/data/base.yaml --backend awscli