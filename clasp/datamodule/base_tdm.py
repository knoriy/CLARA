import os

from abc import ABC, abstractmethod
from typing import Optional

from torchdata.dataloader2 import DataLoader2, DistributedReadingService, MultiProcessingReadingService, SequentialReadingService 
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import logging
pl_logger = logging.getLogger('pytorch_lightning')

class BaseTDM(pl.LightningDataModule, ABC):
	def __init__(self, 
			train_urls:Optional[list]=None,
			test_urls:Optional[list]=None,
			valid_urls:Optional[list]=None,
			predict_urls:Optional[list]=None,
			batch_size:Optional[int]=1,
			num_workers:Optional[int]=0,
			persistent_workers:Optional[bool]=True,
			shuffle:Optional[bool]=True,
			dataloader2:Optional[bool]=False,
			pin_memory:Optional[bool]=True,
			drop_last:Optional[bool]=False,
			exclude_urls:Optional[list]=[],
		):
		super().__init__()

		self.train_data_dir = train_urls
		self.test_data_dir = test_urls
		self.valid_data_dir = valid_urls
		self.predict_data_dir = predict_urls

		self.exclude_urls = exclude_urls

		self.shuffle = shuffle
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.persistent_workers = persistent_workers
		self.pin_memory = pin_memory
		self.drop_last = drop_last
		self.dataloader2 = dataloader2

	@abstractmethod
	def to_sampels(self, data):
		pass
	
	@abstractmethod
	def create_pipeline(self, data_dir):
		pass

	@abstractmethod
	def collate_fn(self, data):
		pass

	def exclude_fn(self, url):
		if url not in self.exclude_urls:
			return True
		return False

	def setup(self, stage:Optional[str] = None):
		if self.train_data_dir and len(self.train_data_dir)>0:
			self.train_datapipe = self.create_pipeline(self.train_data_dir)

		if self.test_data_dir and len(self.test_data_dir)>0:
			self.test_datapipe = self.create_pipeline(self.test_data_dir)

		if self.valid_data_dir and len(self.valid_data_dir)>0:
			self.valid_datapipe = self.create_pipeline(self.valid_data_dir)

		if self.predict_data_dir and len(self.predict_data_dir)>0:
			self.predict_datapipe = self.create_pipeline(self.predict_data_dir)

	def _get_dataloader2(self, dataset):
		service = [
			DistributedReadingService(),
			MultiProcessingReadingService(num_workers=self.num_workers),
		]
		reading_service = SequentialReadingService(*service)
		return DataLoader2(dataset, reading_service=reading_service)
	
	def _get_dataloader(self, dataset, shuffle=None):
		return DataLoader(
			dataset = dataset, 
			num_workers = self.num_workers, 
			batch_size = self.batch_size, 
			collate_fn = self.collate_fn, 
			pin_memory = self.pin_memory, 
			shuffle = shuffle, 
			persistent_workers = self.persistent_workers,
			drop_last = self.drop_last
			)

	def train_dataloader(self):
		if self.dataloader2:
			self.train_dl =  self._get_dataloader2(self.train_datapipe)
		else:
			self.train_dl = self._get_dataloader(self.train_datapipe)
		return self.train_dl

	def val_dataloader(self):
		if self.dataloader2:
			self.valid_dl =  self._get_dataloader2(self.valid_datapipe)
		else:
			self.valid_dl = self._get_dataloader(self.valid_datapipe, shuffle=False)
		return self.valid_dl

	def test_dataloader(self):
		if self.dataloader2:
			self.test_dl =  self._get_dataloader2(self.test_datapipe)
		else:
			self.test_dl = self._get_dataloader(self.test_datapipe, shuffle=False)
		return self.test_dl

	def predict_dataloader(self):
		if self.dataloader2:
			self.valid_dl =  self._get_dataloader2(self.valid_datapipe)
		else:
			self.valid_dl = self._get_dataloader(self.valid_datapipe, shuffle=False)
		return self.valid_dl
