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

from utils import get_s3_paths, get_local_paths, get_lists 
from .utils import get_log_melspec, group_by_filename
from .utils import Boto3FileOpenerIterDataPipe as Boto3FileOpener

import logging
pl_logger = logging.getLogger('pytorch_lightning')

_EMOTION_DICT = {"happy": 0,"sad": 1,"angry": 2,"excited": 3,"sarcastic": 4,"neutral": 5,"disgust": 6,"surprised": 7}
_GENDER_DICT = {"male": 0,"males": 0, "Male": 0, "female": 1, "females": 1, "Female": 1}

class EMNSTDM(BaseTDM):
	def __init__(self,
			root_data_path:str,
			exclude_list:Optional[str]=None,
			cache_path:Optional[str]=None,
			use_cache:Optional[bool]=False,
			recache:Optional[bool]=False,
			train_valid_test:Optional[list]=['train', 'valid', 'test'],
			*args, **kwargs):
		super().__init__(*args, **kwargs)
		self.tokeniser = Tokeniser()

		exclude = []
		if exclude_list:
			exclude = get_lists(exclude_list)
		
		dataset_name = 'EMNS'

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

	def to_samples(self, data):
		a, t = data
		return soundfile.read(io.BytesIO(a[1].read())), json.loads(t[1].read().decode('utf-8'))
	
	def to_keys(self, data):
		audio, labels  = data

		new_labels = {}

		new_labels['text'] = labels['text']
		new_labels['emotion'] = _EMOTION_DICT.get(labels['original_data']['emotion'].lower())
		new_labels['gender'] = _GENDER_DICT.get(labels['original_data']['gender'])

		return audio, new_labels 
	
	def filter_fn(self, sample):
		if _GENDER_DICT.get(sample[1]['gender'], None) and sample[1]['age']:
			return True
		return False

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
			.map(self.to_keys)\

		if self.dataloader2:
			datapipe = datapipe.batch(self.batch_size) \
				.map(self.collate_fn)\
				.fullsync()

		return datapipe

	def collate_fn(self, batch):
		audios, labels = zip(*batch)

		texts = [torch.tensor(self.tokeniser.encode(", ".join(l["text"]))) for l in labels]
		genders = torch.tensor([label['gender'] for label in labels])
		emotion = torch.tensor([label['emotion'] for label in labels])

		mels = [get_log_melspec(a[0], a[1]) for a in audios]
		mel_lengths = [mel.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		
		text_lengths = [text.size(0) for text in texts]
		text_lengths = torch.tensor(text_lengths)

		mels = pad_sequence(mels).permute(1,2,0).contiguous()
		texts = pad_sequence(texts).T.contiguous()

		new_labels = {
			"texts": texts,
			"gender": genders,
			"emotion": emotion,
			}

		return new_labels, mels, text_lengths, mel_lengths