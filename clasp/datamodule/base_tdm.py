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
			batch_size:Optional[int]=1,
			num_workers:Optional[int]=0,
			persistent_workers:Optional[bool]=True,
			shuffle:Optional[bool]=True,
		):
		super().__init__()

		self.train_data_dir = train_urls
		self.test_data_dir = test_urls
		self.valid_data_dir = valid_urls

		self.shuffle = shuffle
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.persistent_workers = persistent_workers

	@abstractmethod
	def to_sampels(self, data):
		pass
	
	@abstractmethod
	def create_pipeline(self, data_dir):
		pass

	@abstractmethod
	def collate_fn(self, data):
		pass

	def setup(self, stage:Optional[str] = None):
		if self.train_data_dir and len(self.train_data_dir)>0:
			self.train = self.create_pipeline(self.train_data_dir)

		if self.test_data_dir and len(self.test_data_dir)>0:
			self.test = self.create_pipeline(self.test_data_dir)

		if self.valid_data_dir and len(self.valid_data_dir)>0:
			self.valid = self.create_pipeline(self.valid_data_dir)

	def _dataloader2(self, dataset):
		service = [
			DistributedReadingService(),
			MultiProcessingReadingService(num_workers=self.num_workers),
		]
		reading_service = SequentialReadingService(*service)
		return DataLoader2(dataset, reading_service=reading_service)

	def _dataloader(self, dataset):
		return DataLoader(dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

	def train_dataloader(self):
		if not self.train_data_dir:
			raise MisconfigurationException('train_data_dir not set.')
		return self._dataloader(self.train)

	def val_dataloader(self):
		if not self.valid_data_dir:
			raise MisconfigurationException('valid_data_dir not set.')
		return self._dataloader(self.valid)

	def test_dataloader(self):
		if not self.test_data_dir:
			raise MisconfigurationException('test_data_dir not set.')
		return self._dataloader(self.test)

	def predict_dataloader(self):
		if not self.test_data_dir:
			raise MisconfigurationException('test_data_dir not set.')
		return self._dataloader(self.test)