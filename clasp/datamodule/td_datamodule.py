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

from torch.utils.data import DataLoader

import logging
pl_logger = logging.getLogger('pytorch_lightning')

from text.whisper.normalizers import EnglishTextNormalizer
from text.tokeniser import Tokeniser # from text.whisper.tokenizer import get_tokenizer
from utils import get_s3_paths, get_local_paths, get_lists 

class MultilingualTDM(pl.LightningDataModule):
	def __init__(self, 
			root_data_path:str,#'s-laion-audio/webdataset_tar/' or '/fsx/knoriy/processed_datasets/', 
			dataset_list:str,
			exclude_list:Optional[str]=None,
			batch_size:Optional[int]=None,
			num_workers:Optional[int]=0,
			persistent_workers:Optional[bool]=True,
			shuffle:Optional[bool]=True,
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
		
		if root_data_path.startswith('s3://'):
			root_data_path = root_data_path.replace('s3://', '')
			urls = get_s3_paths(
				base_path			= root_data_path,
				train_valid_test	= ['train', 'test', 'valid'],
				dataset_names		= dataset_names, 
				exclude				= exclude,
				cache_path			= f"./tmp/s3_{os.path.basename(dataset_list)}.json",
				use_cache			= True,
				recache				= True,
				)
		else:
			urls = get_local_paths(
				base_path			= root_data_path,
				train_valid_test	= ['train', 'test', 'valid'],
				dataset_names		= dataset_names, 
				exclude				= exclude,
				cache_path			= f"./tmp/local_{os.path.basename(dataset_list)}.json",
				use_cache			= True
				)

		pl_logger.info(f"Urls found: \
			\n\tTrain: {len(urls['train'])} \
			\n\tValid: {len(urls['valid'])} \
			\n\tTest: {len(urls['test'])}"
		)

		assert urls['train'], "Train URLs is empty"
		assert urls['valid'], "Valid URLs is empty"
		assert urls['test'], "Test URLs is empty"

		self.train_data_dir = urls['train']
		self.test_data_dir = urls['test']
		self.valid_data_dir = urls['valid']

		self.shuffle = shuffle
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.persistent_workers = persistent_workers

		self.cleaner = EnglishTextNormalizer()
		self.tokenizer = Tokeniser() # self.tokenizer = get_tokenizer(True)

	def to_sampels(self, data):
		a, t = data
		return soundfile.read(io.BytesIO(a[1].read())), json.loads(t[1].read().decode('utf-8'))
	
	def _create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.shuffle()\
			.sharding_filter()\
			.open_files_by_fsspec(mode='rb')\
			.load_from_tar() \
			.batch(2) \
			.shuffle(buffer_size=self.batch_size)\
			.map(self.to_sampels) \
			.batch(self.batch_size) \
			.map(self.collate_fn)
		
		return datapipe

	def setup(self, stage:Optional[str] = None):
		if len(self.train_data_dir)>0:
			self.train = self._create_pipeline(self.train_data_dir)

		if len(self.test_data_dir)>0:
			self.test = self._create_pipeline(self.test_data_dir)

		if len(self.valid_data_dir)>0:
			self.valid = self._create_pipeline(self.valid_data_dir)

	def _dataloader2(self, dataset):
		service = [
			# DistributedReadingService(),
			MultiProcessingReadingService(num_workers=self.num_workers),
		]
		reading_service = SequentialReadingService(*service)
		return DataLoader2(dataset, reading_service=reading_service)
	
	def _dataloader(self, dataset):
		return DataLoader(dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

	def train_dataloader(self):
		self.train_dl = self._dataloader2(self.train)
		return self.train_dl

	# def val_dataloader(self):
	# 	self.val_dl = self._dataloader2(self.valid)
	# 	return self.val_dl

	def test_dataloader(self):
		self.test_dl = self._dataloader2(self.test)
		return self.test_dl

	def predict_dataloader(self):
		self.predict_dl = self._dataloader2(self.valid)
		return self.predict_dl
	
	def tokeniser_encode(self, text:str, lanuage:str='en'):
		return self.tokenizer.encode(self.cleaner(text), language=lanuage)
	
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
			dataset_list='/fsx/knoriy/code/CLASP/config/test_list.txt',
			exclude_list='/fsx/knoriy/code/CLASP/config/exclude_list.txt',
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