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

_SOUNDS_DICT = {
    "dog": 0,
    "rooster": 1,
    "pig": 2,
    "cow": 3,
    "frog": 4,
    "cat": 5,
    "hen": 6,
    "insects": 7,
    "sheep": 8,
    "crow": 9,
    "rain": 10,
    "sea waves": 11,
    "crackling fire": 12,
    "crickets": 13,
    "chirping birds": 14,
    "water drops": 15,
    "wind": 16,
    "pouring water": 17,
    "toilet flush": 18,
    "thunderstorm": 19,
    "crying baby": 20,
    "sneezing": 21,
    "clapping": 22,
    "breathing": 23,
    "coughing": 24,
    "footsteps": 25,
    "laughing": 26,
    "brushing teeth": 27,
    "snoring": 28,
    "drinking sipping": 29,
    "door wood knock": 30,
    "mouse click": 31,
    "keyboard typing": 32,
    "door wood creaks": 33,
    "can opening": 34,
    "washing machine": 35,
    "vacuum cleaner": 36,
    "clock alarm": 37,
    "clock tick": 38,
    "glass breaking": 39,
    "helicopter": 40,
    "chainsaw": 41,
    "siren": 42,
    "car horn": 43,
    "engine": 44,
    "train": 45,
    "church bells": 46,
    "airplane": 47,
    "fireworks": 48,
    "hand saw": 49
}

class ESC50TDM(BaseTDM):
	def __init__(self, 
	      	root_data_path:str,
			classes:str=None, 
			exclude_list:Optional[str]=None,
			cache_path:Optional[str]=None,
			use_cache:Optional[bool]=False,
			recache:Optional[bool]=False,
			train_valid_test:Optional[list]=['train', 'valid', 'test'],
			*args, **kwargs):
		super().__init__(*args, **kwargs)
		self.tokeniser = Tokeniser()
		self.classes = classes
		if not self.classes:
			self.classes = _SOUNDS_DICT

		exclude = []
		if exclude_list:
			exclude = get_lists(exclude_list)
		
		dataset_name = 'esc50'

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

		classes = self.classes.get(labels["original_data"]["category"])

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
		# WARMING: ONLY USING THE FIRST LABEL
		classes = torch.tensor([l['labels'] for l in labels])
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