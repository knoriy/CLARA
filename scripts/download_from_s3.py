import sys
sys.path.append('/fsx/knoriy/code/CLASP/clasp')


import torchdata
from torch.utils.data import DataLoader
import tqdm

from utils import get_s3_paths, get_lists
from datamodule.utils import Boto3FileOpenerIterDataPipe as Boto3FileOpener
from datamodule.utils import filepath_fn


def main():
    dataset_names = get_lists('/fsx/knoriy/code/CLASP/config/dataset_list.txt')
    exclude_names = get_lists('/fsx/knoriy/code/CLASP/config/exclude_list.txt')
    root_data_path = 's3://s-laion-audio/webdataset_tar/'


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
        .on_disk_cache(filepath_fn=filepath_fn)\
        .open_files_by_boto3(mode='rb')\
        .end_caching(filepath_fn=filepath_fn)\
        
    dl = DataLoader(datapipe, num_workers=48, persistent_workers=True)
    
    for _ in tqdm.tqdm(dl, total=len(updated_urls), desc='Downloading'):
        pass

if __name__ == '__main__':
    main()