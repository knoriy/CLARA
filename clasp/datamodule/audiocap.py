from typing import Optional

import io
import json
import soundfile

import torch
import torchdata
from torchdata.datapipes.iter import FSSpecFileOpener
from torch.nn.utils.rnn import pad_sequence

# from .base_tdm import BaseTDM
from datamodule.base_tdm import BaseTDM
from text.tokeniser import Tokeniser # from text.whisper.tokenizer import get_tokenizer
from .utils import Boto3FileOpenerIterDataPipe as Boto3FileOpener
from .utils import Boto3FileListerIterDataPipe as Boto3FileLister
from .utils import get_log_melspec, group_by_filename

from utils import get_s3_paths, get_local_paths, get_lists 

import logging
pl_logger = logging.getLogger('pytorch_lightning')

class AudioCapTDM(BaseTDM):
	def __init__(self, 
	      	root_data_path:str,
			exclude_list:Optional[str]=None,
			cache_path:Optional[str]=None,
			use_cache:Optional[bool]=True,
			recache:Optional[bool]=False,
			train_valid_test:Optional[list]=['train', 'valid', 'test'],
			*args, **kwargs):
		super().__init__(*args, **kwargs)
		self.tokeniser = Tokeniser()

		exclude = []
		if exclude_list:
			exclude = get_lists(exclude_list)
		
		dataset_name = 'audiocaps'

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

		self.train_data_dir = self.urls.get('train', None)
		self.test_data_dir = self.urls.get('test', None)
		self.valid_data_dir = self.urls.get('valid', None)

		pl_logger.info(f"Urls found: \
			\n\tTrain: {len(self.urls.get('train', None))} \
			\n\tValid: {len(self.urls.get('valid', None))} \
			\n\tTest: {len(self.urls.get('test', None))}"
		)

		self.tokenizer = Tokeniser()

	def tokeniser_encode(self, text:str, lanuage:str='en'):
		return self.tokenizer.encode(text, language=lanuage)
	
	def tokeniser_decode(self, tensor:torch.Tensor):
		return self.tokenizer.decode(tensor)


	def to_samples(self, data):
		a, t = data
		return soundfile.read(io.BytesIO(a[1].read())), json.loads(t[1].read().decode('utf-8'))
	
	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.shuffle()\
			.sharding_filter()
			
		if self.is_local:
			datapipe = FSSpecFileOpener(datapipe, mode='rb')
		else:
			datapipe = Boto3FileOpener(datapipe, mode='rb')

		datapipe = datapipe.load_from_tar() \
			.batch(2) \
			.shuffle(buffer_size=100)\
			.map(self.to_samples) \

		if self.dataloader2:
			datapipe = datapipe.batch(self.batch_size) \
				.map(self.collate_fn)\
				.fullsync()

		return datapipe

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

		return texts, mels, text_lengths, mel_lengths
	