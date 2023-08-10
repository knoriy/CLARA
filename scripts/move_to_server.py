import os
import sys
sys.path.append("/fsx/knoriy/CLASP/clasp/")

import logging
import torchdata
import paramiko

from typing import Optional
from torchdata.datapipes.iter import FSSpecFileOpener

from datamodule.td_datamodule import BaseTDM
from utils import get_s3_paths, get_local_paths, get_lists
from datamodule.utils import Boto3FileOpenerIterDataPipe as Boto3FileOpener

pl_logger = logging.getLogger('pytorch_lightning')

class ToHFHub(BaseTDM):
	def __init__(self, 
			root_data_path:str,#'s-laion-audio/webdataset_tar/' or '/fsx/knoriy/processed_datasets/', 
			dataset_list:str,
			hostname:str,
			username:str,
			save_path:str='/data/',
			exclude_list:Optional[str]=None,
			cache_path:Optional[str]=None,
			use_cache:Optional[bool]=False,
			recache:Optional[bool]=False,
			train_valid_test:Optional[list]=['train', 'valid', 'test'],
			*args,**kwargs,
		):
		super().__init__(*args,**kwargs)

		self.save_path = save_path

		exclude = []
		if exclude_list:
			exclude = get_lists(exclude_list)

		dataset_names = get_lists(dataset_list)
		nl = "\n\t"
		pl_logger.info(f"Dataset names:{nl}{nl.join(map(str ,dataset_names))}")

		dataset_names_intersection = set(dataset_names).intersection(exclude)
		if dataset_names_intersection:
			raise Warning(f'Found similary dataset names in datasets and excluded dataset: {dataset_names_intersection}')
		
		if not cache_path:
			cache_path = f"logs/{os.path.basename(dataset_list)}.json"
		
		if root_data_path.startswith('s3://'):
			self.is_local = False
			root_data_path = root_data_path.replace('s3://', '')
			self.urls = get_s3_paths(
				base_path			= root_data_path,
				train_valid_test	= train_valid_test,
				dataset_names		= dataset_names, 
				exclude				= exclude,
				cache_path			= cache_path,
				use_cache			= use_cache,
				recache				= recache
				)
		else:
			self.is_local = True
			self.urls = get_local_paths(
				base_path			= root_data_path,
				train_valid_test	= train_valid_test,
				dataset_names		= dataset_names, 
				exclude				= exclude,
				cache_path			= cache_path,
				use_cache			= use_cache,
				recache				= recache
				)

		pl_logger.info(f"Urls found: \
			\n\tTrain: {len(self.urls.get(train_valid_test[0], []))} \
			\n\tValid: {len(self.urls.get(train_valid_test[1], []))} \
			\n\tTest: {len(self.urls.get (train_valid_test[2], []))}"
		)
		
		self.train_data_dir = self.urls.get(train_valid_test[0], None)
		self.valid_data_dir = self.urls.get(train_valid_test[1], None)
		self.test_data_dir = self.urls.get(train_valid_test[2], None)

		private_key = paramiko.Ed25519Key.from_private_key_file(os.path.expanduser('~/.ssh/id_ed25519'))
		self.ssh = paramiko.SSHClient() 
		self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		self.ssh.connect(hostname, username=username, pkey=private_key)

	def to_samples(self, data):
		return data

	
	def mkdir_p(self, sftp, remote_directory):
		if remote_directory == '/':
			sftp.chdir('/')
			return
		if remote_directory == '':
			return
		
		try:
			sftp.chdir(remote_directory)
		except IOError:
			dirname, basename = os.path.split(remote_directory.rstrip('/'))
			self.mkdir_p(sftp, dirname)
			sftp.mkdir(basename)
			sftp.chdir(basename)
			return True

	def to_hf(self, data):
		path, content_stream = data
		sftp = self.ssh.open_sftp()

		destination_dir = os.path.join(self.save_path, *path.split('/')[4:])
		self.mkdir_p(sftp, os.path.dirname(destination_dir))
		try:
			sftp.stat(destination_dir) 
			print(f"{destination_dir} already exists on remote") 
		except FileNotFoundError: 
			sftp.putfo(content_stream, destination_dir)
		sftp.close()
		return path
	
	def collate_fn(self, data):
		return data
	
	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.sharding_filter() # Sharding filter here causes the dataloader to hang when using multiple GPUs
		
		if self.is_local:
			datapipe = FSSpecFileOpener(datapipe, mode='rb')
		else:
			datapipe = Boto3FileOpener(datapipe, mode='rb')
		
		datapipe = datapipe.map(self.to_hf)
		
		return datapipe


if __name__ == '__main__':
	import tqdm
	dp = ToHFHub(
		root_data_path='s3://laion-west-audio/webdataset_tar/', 
		dataset_list='/fsx/knoriy/CLASP/logs/test_list.txt',
		hostname='65.109.157.234',
		username='root',
	)
	dp.setup()

	if hasattr(dp, 'test_datapipe'):
		for i in tqdm.tqdm(dp.test_datapipe, total=len(dp.test_data_dir)):
			print(i)
	if hasattr(dp, 'valid_datapipe'):
		for i in tqdm.tqdm(dp.valid_datapipe, total=len(dp.valid_data_dir)):
			print(i)
	if hasattr(dp, 'train_datapipe'):
		for i in tqdm.tqdm(dp.train_datapipe, total=len(dp.train_data_dir)):
			print(i)
	