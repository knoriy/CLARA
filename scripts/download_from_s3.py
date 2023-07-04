import os
import sys
sys.path.append("/fsx/knoriy/CLASP/clasp/")
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple, TypeVar


import torchdata
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from torch.utils.data import DataLoader
import tqdm

from utils import get_s3_paths, get_lists
from datamodule.utils import Boto3FileOpenerIterDataPipe as Boto3FileOpener
from datamodule.utils import filepath_fn

from torchdata.datapipes.iter.util.cacheholder import CacheState, _get_list_filename, _hash_check, _promise_filename
from torchdata.datapipes.iter.util.cacheholder import OnDiskCacheHolderIterDataPipe as OnDiskCacheHolder
try:
    import portalocker
except ImportError:
    portalocker = None


class OnDiskCacheHolderIterDataPipe(OnDiskCacheHolder):
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

def main():
    dataset_names = get_lists('/fsx/knoriy/CLASP/config/dataset_list.txt')
    exclude_names = get_lists('/fsx/knoriy/CLASP/config/exclude_list.txt')
    root_data_path = 's3://laion-west-audio/webdataset_tar/'


    root_data_path = root_data_path.replace('s3://', '')
    urls = get_s3_paths(
        base_path			= root_data_path,
        train_valid_test	= {"train", "valid"},
        dataset_names		= dataset_names, 
        exclude				= exclude_names,
        recache				= True
        )
    
    updated_urls = []
    for key in urls.keys():
        updated_urls.extend(urls[key])
    
    datapipe = torchdata.datapipes.iter.IterableWrapper(updated_urls)\
        .sharding_filter()\

    datapipe = OnDiskCacheHolderIterDataPipe(datapipe, filepath_fn=filepath_fn)\
        .open_files_by_boto3(mode='rb')\
        .end_caching(filepath_fn=filepath_fn)\
        
    dl = DataLoader(datapipe, num_workers=12, persistent_workers=True)
    
    for _ in tqdm.tqdm(dl, total=len(updated_urls), desc='Downloading'):
        pass

if __name__ == '__main__':
    main()