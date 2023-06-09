import os
import tempfile
import librosa
import numpy as np
import torch
import glob

def get_log_melspec(audio, sr, n_mels=80, n_fft=1024, hop_length=512, win_length=1024, fmin=0, fmax=8000, **kwargs):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, win_length=win_length, hop_length=hop_length, fmin=fmin, fmax=fmax, **kwargs)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel+40)/40
    return torch.tensor(mel, dtype=torch.float32).T

def filepath_fn(url):
	temp_dir = tempfile.gettempdir()
	_, dataset_name, folder_name, file_name = url.rsplit('/', 3)
	return os.path.join(temp_dir, "CLASP", dataset_name, folder_name, file_name)

def delete_primise_fn(url):
	files = glob.glob(f"{url}.*")
	for file in files:
		os.remove(file)
	return url

def delete_cache_fn(cache_path):
    tar_path = os.path.dirname(cache_path[0])
    if os.path.isfile(tar_path):
        os.remove(tar_path)
    return cache_path


def group_by_filename(x):
	return os.path.basename(x[0]).split(".")[0]



from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from torch.utils.data.datapipes.utils.common import match_masks

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from torchdata.datapipes.utils import StreamWrapper

import boto3
from io import BytesIO

U = Union[bytes, bytearray, str]


def _assert_boto3() -> None:
    try:
        import boto3
    except ImportError:
        raise ModuleNotFoundError(
            "Package `boto3` is required to be installed to use this datapipe."
            "Please use `pip install boto3` to install the package"
        )


@functional_datapipe("list_files_by_boto3")
class Boto3FileListerIterDataPipe(IterDataPipe[str]):
    r"""
    Lists the contents of the directory at the provided ``root`` pathname or URL,
    and yields the full pathname or URL for each file within the
    directory (functional name: ``list_files_by_boto3``).

    Args:
        root: The root S3 bucket name or list of bucket names to list files from
        masks: Unix style filter string or string list for filtering file name(s)
        kwargs: Extra options that make sense to a particular S3 connection,
            e.g. region_name, aws_access_key_id, aws_secret_access_key, etc.

    Example:

    .. testsetup::

        bucket_name = "my-bucket"

    .. testcode::

        from torchdata.datapipes.iter import Boto3FileLister

        datapipe = Boto3FileLister(root=bucket_name)
    """

    def __init__(
        self,
        root: Union[str, Sequence[str], IterDataPipe],
        masks: Union[str, List[str]] = "",
        **kwargs,
    ) -> None:
        _assert_boto3()

        if isinstance(root, str):
            root = [
                root,
            ]
        if not isinstance(root, IterDataPipe):
            self.datapipe: IterDataPipe = IterableWrapper(root)  # type: ignore[assignment]
        else:
            self.datapipe = root
        self.masks = masks
        self.kwargs_for_connection = kwargs

    def __iter__(self) -> Iterator[str]:
        for root in self.datapipe:
            s3 = boto3.client('s3', **self.kwargs_for_connection)
            bucket, prefix = root.replace('s3://', '').split('/', 1)
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            contents = response.get('Contents', [])
            for content in contents:
                file_name = content['Key']
                if not match_masks(file_name, self.masks):
                    continue
                yield f"s3://{bucket}/{file_name}"


@functional_datapipe("open_files_by_boto3")
class Boto3FileOpenerIterDataPipe(IterDataPipe[Tuple[str, BytesIO]]):
    r"""
    Opens files from input datapipe which contains S3 paths and yields a tuple of
    pathname and opened file stream (functional name: ``open_files_by_boto3``).

    Args:
        source_datapipe: Iterable DataPipe that provides the S3 paths
        mode: An optional string that specifies the mode in which the file is opened (``"r"`` by default)
        kwargs_for_open: Optional Dict to specify kwargs for opening files
        kwargs: Extra options that are used to establish the S3 connection,
            e.g. region_name, aws_access_key_id, aws_secret_access_key, etc.

    Example:

    .. testsetup::

        bucket_name = "my-bucket"

    .. testcode::

        from torchdata.datapipes.iter import Boto3FileLister

        datapipe = Boto3FileLister(root=bucket_name)
        file_dp = datapipe.open_files_by_boto3()
    """

    def __init__(
        self, source_datapipe: IterDataPipe[str], mode: str = "r", *, kwargs_for_open: Optional[Dict] = None, **kwargs
    ) -> None:
        _assert_boto3()

        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.mode: str = mode
        self.kwargs_for_open = kwargs_for_open if kwargs_for_open is not None else {}
        self.kwargs_for_connection = kwargs

    def __iter__(self) -> Iterator[Tuple[str, BytesIO]]:
        for file_uri in self.source_datapipe:
            bucket, key = file_uri[5:].split('/', 1)
            s3 = boto3.client('s3', **self.kwargs_for_connection)
            response = s3.get_object(Bucket=bucket, Key=key)
            file_content = response['Body'].read()
            yield file_uri, StreamWrapper(BytesIO(file_content))

    def __len__(self) -> int:
        return len(self.source_datapipe)