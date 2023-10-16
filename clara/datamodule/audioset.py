from typing import Optional

import io
import json
import soundfile

import torch
import torchdata
from torchdata.datapipes.iter import FSSpecFileOpener
from torch.nn.utils.rnn import pad_sequence

# from .base_tdm import BaseTDM
from .base_tdm import BaseTDM
from .text.tokeniser import Tokeniser # from text.whisper.tokenizer import get_tokenizer
from .utils import Boto3FileOpenerIterDataPipe as Boto3FileOpener
from .utils import Boto3FileListerIterDataPipe as Boto3FileLister
from .utils import get_log_melspec, group_by_filename

from utils import get_s3_paths, get_local_paths, get_lists 

import logging
pl_logger = logging.getLogger('pytorch_lightning')

class AudioSetTDM(BaseTDM):
	def __init__(self, 
	      	root_data_path:str,
			classes:str, 
			exclude_list:Optional[str]=None,
			cache_path:Optional[str]=None,
			use_cache:Optional[bool]=False,
			recache:Optional[bool]=False,
			train_valid_test:Optional[list]=['unbalanced_train', 'valid', 'test'],
			*args, **kwargs):
		super().__init__(*args, **kwargs)
		self.tokeniser = Tokeniser()
		self.classes = classes

		exclude = []
		if exclude_list:
			exclude = get_lists(exclude_list)
		
		dataset_name = 'audioset'

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

		self.train_data_dir = self.urls.get(train_valid_test[0], None)
		self.test_data_dir = self.urls.get(train_valid_test[2], None)
		self.valid_data_dir = self.urls.get(train_valid_test[1], None)

		pl_logger.info(f"Urls found: \
			\n\tTrain: {len(self.urls.get(train_valid_test[0], []))} \
			\n\tValid: {len(self.urls.get(train_valid_test[1], []))} \
			\n\tTest: {len(self.urls.get(train_valid_test[2], []))}"
		)

	def to_samples(self, data):
		a, t = data
		return soundfile.read(io.BytesIO(a[1].read())), json.loads(t[1].read().decode('utf-8'))
	
	def to_keys(self, data):
		audio, labels  = data

		classes = [self.classes.get(l) for l in labels["original_data"]["class_names"]]

		new_labels = {
			"text": labels["text"],
			"labels": classes,
		}

		return audio, new_labels 

	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.shuffle()\
			.sharding_filter()
			
		if self.is_local:
			datapipe = FSSpecFileOpener(datapipe, mode='rb')
		else:
			datapipe = Boto3FileLister(datapipe, masks=["*.tar"])
			datapipe = Boto3FileOpener(datapipe, mode='rb')

		datapipe = datapipe.load_from_tar() \
			.batch(2) \
			.shuffle(buffer_size=100)\
			.map(self.to_samples) \
			.map(self.to_keys)\

		if self.dataloader2:
			datapipe = datapipe.batch(self.batch_size) \
				.map(self.collate_fn)\
				.fullsync()

		return datapipe

	def collate_fn(self, batch):
		audios, labels = zip(*batch)
		# WARNING: ONLY USING THE FIRST LABEL
		classes = torch.tensor([l['labels'][0] for l in labels])
		texts = [torch.tensor(self.tokeniser.encode(", ".join(l["text"]))) for l in labels]

		mels = [get_log_melspec(a[0], a[1]) for a in audios]

		mel_lengths = [mel.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		text_lengths = [text.size(0) for text in texts]
		text_lengths = torch.tensor(text_lengths)

		mels = pad_sequence(mels).permute(1,2,0).contiguous()
		texts = pad_sequence(texts).T.contiguous()

		new_labels = {
			"texts": texts,
			"sounds": classes,
			}

		return new_labels, mels, text_lengths, mel_lengths