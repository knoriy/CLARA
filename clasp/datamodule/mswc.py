import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import json
import soundfile

from typing import Optional

import torch
import torchdata
from torch.nn.utils.rnn import pad_sequence
from torchdata.datapipes.iter import FileOpener, FSSpecFileOpener

import logging
pl_logger = logging.getLogger('pytorch_lightning')

from text.tokeniser import Tokeniser # from text.whisper.tokenizer import get_tokenizer
from utils import get_s3_paths, get_local_paths, get_lists
from .utils import Boto3FileOpenerIterDataPipe as Boto3FileOpener
from .utils import get_log_melspec
from . import BaseTDM


class MSWCTDM(BaseTDM):
	def __init__(self, 
			root_data_path:str,#'s-laion-audio/webdataset_tar/' or '/fsx/knoriy/processed_datasets/', 
			exclude_list:Optional[str]=None,
			cache_path:Optional[str]=None,
			use_cache:Optional[bool]=False,
			recache:Optional[bool]=False,
			train_valid_test:Optional[list]=['train', 'valid', 'test'],
			*args,**kwargs,
		):
		super().__init__(*args,**kwargs)

		exclude = []
		if exclude_list:
			exclude = get_lists(exclude_list)

		dataset_name = 'mswc'

		dataset_names_intersection = set(dataset_name).intersection(exclude)
		if dataset_names_intersection:
			raise Warning(f'Found similary dataset names in datasets and excluded dataset: {dataset_names_intersection}')
		
		if not cache_path:
			cache_path = f"logs/{dataset_name}.json"
		
		if root_data_path.startswith('s3://'):
			self.is_local = False
			root_data_path = root_data_path.replace('s3://', '')
			self.urls = get_s3_paths(
				base_path			= root_data_path,
				train_valid_test	= train_valid_test,
				dataset_names		= dataset_name, 
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
				dataset_names		= dataset_name, 
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

		self.tokenizer = Tokeniser()

	def to_samples(self, data):
		a, t = data
		return soundfile.read(io.BytesIO(a[1].read())), json.loads(t[1].read().decode('utf-8'))
	
	def create_pipeline(self, data_dir):
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
			.shuffle()\
			.map(self.to_samples) \
		
		if self.dataloader2:
			datapipe = datapipe.batch(self.batch_size) \
				.map(self.collate_fn)\
				.fullsync()
		
		return datapipe

	def tokeniser_encode(self, text:str, lanuage:str='en'):
		return self.tokenizer.encode(text, language=lanuage)
	
	def tokeniser_decode(self, tensor:torch.Tensor):
		return self.tokenizer.decode(tensor)
	
	def collate_fn(self, data):
		mels = []
		texts = []

		for (a, t) in data:
			mels.append(get_log_melspec(a[0], a[1]))

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
		
		new_labels = {
			"texts": texts,
		}

		return new_labels, mels, text_lengths, mel_lengths