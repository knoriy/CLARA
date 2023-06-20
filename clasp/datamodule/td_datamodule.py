import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import json
import librosa
import soundfile
import numpy as np

from typing import Optional

import torch
import torchdata
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from torchdata.dataloader2 import DataLoader2, DistributedReadingService, MultiProcessingReadingService, SequentialReadingService 
from torchdata.datapipes.iter import FileOpener, FSSpecFileOpener

from torch.utils.data import DataLoader

import logging
pl_logger = logging.getLogger('pytorch_lightning')

from text.whisper.normalizers import EnglishTextNormalizer
from text.tokeniser import Tokeniser # from text.whisper.tokenizer import get_tokenizer
from utils import get_s3_paths, get_local_paths, get_lists 
from .utils import Boto3FileOpenerIterDataPipe as Boto3FileOpener

class MultilingualTDM(pl.LightningDataModule):
	def __init__(self, 
			root_data_path:str,#'s-laion-audio/webdataset_tar/' or '/fsx/knoriy/processed_datasets/', 
			dataset_list:str,
			exclude_list:Optional[str]=None,
			batch_size:Optional[int]=1,
			num_workers:Optional[int]=0,
			persistent_workers:Optional[bool]=True,
			shuffle:Optional[bool]=True,
			cache_path:Optional[str]=None,
			use_cache:Optional[bool]=True,
			recache:Optional[bool]=False,
			train_valid_test:Optional[list]=['train', 'valid', 'test'],
			dataloader2:Optional[bool]=False,
        ):
		super().__init__()

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
			cache_path = f".logs/{os.path.basename(dataset_list)}.json"
		
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
			\n\tTrain: {len(self.urls.get('train', None))} \
			\n\tValid: {len(self.urls.get('valid', None))} \
			\n\tTest: {len(self.urls.get('test', None))}"
		)
		
		self.train_data_dir = self.urls.get('train', None)
		self.test_data_dir = self.urls.get('test', None)
		self.valid_data_dir = self.urls.get('valid', None)

		self.shuffle = shuffle
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.persistent_workers = persistent_workers
		self.dataloader2 = dataloader2

		# self.cleaner = EnglishTextNormalizer()
		self.tokenizer = Tokeniser()

	def to_sampels(self, data):
		a, t = data
		return soundfile.read(io.BytesIO(a[1].read())), json.loads(t[1].read().decode('utf-8'))
	
	def _create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.shuffle()\
			# .sharding_filter() # Sharding filter here causes the dataloader to hang when using multiple GPUs
		
		if self.is_local:
			datapipe = FSSpecFileOpener(datapipe, mode='rb')
		else:
			datapipe = Boto3FileOpener(datapipe, mode='rb')

		datapipe = datapipe.load_from_tar() \
			.batch(2) \
			.sharding_filter()\
			.shuffle(buffer_size=100)\
			.map(self.to_sampels) \
		
		if self.dataloader2:
			datapipe = datapipe.batch(self.batch_size) \
				.map(self.collate_fn)\
				.fullsync()
		
		return datapipe

	def setup(self, stage:Optional[str] = None):
		if len(self.train_data_dir)>0:
			self.train_datapipe = self._create_pipeline(self.train_data_dir)

		if len(self.test_data_dir)>0:
			self.test_datapipe = self._create_pipeline(self.test_data_dir)

		if len(self.valid_data_dir)>0:
			self.valid_datapipe = self._create_pipeline(self.valid_data_dir)

	def _get_dataloader2(self, dataset):
		service = [
			DistributedReadingService(),
			MultiProcessingReadingService(num_workers=self.num_workers),
		]
		reading_service = SequentialReadingService(*service)
		return DataLoader2(dataset, reading_service=reading_service)
	
	def _get_dataloader(self, dataset):
		return DataLoader(dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn, pin_memory=True)

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
			self.valid_dl = self._get_dataloader(self.valid_datapipe)
		return self.valid_dl

	def test_dataloader(self):
		if self.dataloader2:
			self.test_dl =  self._get_dataloader2(self.test_datapipe)
		else:
			self.test_dl = self._get_dataloader(self.test_datapipe)
		return self.test_dl

	def predict_dataloader(self):
		if self.dataloader2:
			self.valid_dl =  self._get_dataloader2(self.valid_datapipe)
		else:
			self.valid_dl = self._get_dataloader(self.valid_datapipe)
		return self.valid_dl
	
	def tokeniser_encode(self, text:str, lanuage:str='en'):
		return self.tokenizer.encode(text, language=lanuage)
	
	def tokeniser_decode(self, tensor:torch.Tensor):
		return self.tokenizer.decode(tensor)
	
	def collate_fn(self, data):
		mels = []
		texts = []

		for (a, t) in data:
			mel = librosa.feature.melspectrogram(y=a[0], sr=a[1], fmin=0, fmax=8000, n_mels=80, n_fft=1024, win_length=1024, hop_length=512)
			mel = librosa.power_to_db(mel, ref=np.max)
			mel = (mel+40)/40
			mels.append(torch.tensor(mel, dtype=torch.float32).T)

			texts.append(
				torch.tensor(
					self.tokeniser_encode(
						', '.join(t['text']) if isinstance(t['text'], list) else t['text'], \
							'en' if 'original_data' not in t.keys() else t["original_data"]["language"] if "language" in t["original_data"].keys() else 'en'
					)
				) 
			)

		mel_lengths = [mel.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		text_lengths = [text.shape[0] for text in texts]
		text_lengths = torch.tensor(text_lengths)

		texts = pad_sequence(texts).T.contiguous()
		mels = pad_sequence(mels).permute(1,2,0).contiguous()

		return texts, mels, text_lengths, mel_lengths


if __name__ == '__main__':
	import tqdm

	dataset = MultilingualTDM(
			root_data_path='s3://s-laion-audio/webdataset_tar/', 
			dataset_list='./config/test_list.txt',
			exclude_list='./config/exclude_list.txt',
			batch_size = 64,
			num_workers=0,
		)

	dataset.setup()

	for epoch in tqdm.trange(2, desc="train"):
		for i in tqdm.tqdm(dataset.train, desc="train minibatch"):
			print(i)
			breakpoint()
			break
		break
	# for epoch in tqdm.trange(2, desc="valid"):
	# 	for i in tqdm.tqdm(dataset.val_dataloader(), desc="valid minibatch"):
	# 		pass

	# for epoch in tqdm.trange(2, desc="test"):
	# 	for i in tqdm.tqdm(dataset.test_dataloader(), desc="test minibatch"):
	# 		pass