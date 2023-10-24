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

LANGUAGES = {
    "ab": 0,
    "ace": 1,
    "ady": 2,
    "af": 3,
    "am": 4,
    "an": 5,
    "ar": 6,
    "arn": 7,
    "as": 8,
    "ast": 9,
    "az": 10,
    "ba": 11,
    "bas": 12,
    "be": 13,
    "bg": 14,
    "bn": 15,
    "br": 16,
    "bs": 17,
    "bxr": 18,
    "ca": 19,
    "cak": 20,
    "ckb": 21,
    "cnh": 22,
    "co": 23,
    "cs": 24,
    "cv": 25,
    "cy": 26,
    "da": 27,
    "de": 28,
    "dsb": 29,
    "dv": 30,
    "dyu": 31,
    "el": 32,
    "en": 33,
    "eo": 34,
    "es": 35,
    "et": 36,
    "eu": 37,
    "fa": 38,
    "ff": 39,
    "fi": 40,
    "fo": 41,
    "fr": 42,
    "fy-NL": 43,
    "ga-IE": 44,
    "gl": 45,
    "gn": 46,
    "gom": 47,
    "ha": 48,
    "he": 49,
    "hi": 50,
    "hil": 51,
    "hr": 52,
    "hsb": 53,
    "ht": 54,
    "hu": 55,
    "hy-AM": 56,
    "hyw": 57,
    "ia": 58,
    "id": 59,
    "ie": 60,
    "ig": 61,
    "is": 62,
    "it": 63,
    "izh": 64,
    "ja": 65,
    "jbo": 66,
    "ka": 67,
    "kaa": 68,
    "kab": 69,
    "kbd": 70,
    "ki": 71,
    "kk": 72,
    "km": 73,
    "kmr": 74,
    "kn": 75,
    "knn": 76,
    "ko": 77,
    "kpv": 78,
    "kw": 79,
    "ky": 80,
    "lb": 81,
    "lg": 82,
    "lij": 83,
    "ln": 84,
    "lo": 85,
    "lt": 86,
    "lv": 87,
    "mai": 88,
    "mdf": 89,
    "mg": 90,
    "mhr": 91,
    "mk": 92,
    "ml": 93,
    "mn": 94,
    "mni": 95,
    "mos": 96,
    "mr": 97,
    "mrj": 98,
    "ms": 99,
    "mt": 100,
    "my": 101,
    "myv": 102,
    "nan-tw": 103,
    "nb-NO": 104,
    "nd": 105,
    "ne-NP": 106,
    "nia": 107,
    "nl": 108,
    "nn-NO": 109,
    "nr": 110,
    "nso": 111,
    "nyn": 112,
    "oc": 113,
    "om": 114,
    "or": 115,
    "pa-IN": 116,
    "pap-AW": 117,
    "pl": 118,
    "ps": 119,
    "pt": 120,
    "quc": 121,
    "quy": 122,
    "rm-sursilv": 123,
    "rm-vallader": 124,
    "ro": 125,
    "ru": 126,
    "rw": 127,
    "sah": 128,
    "sat": 129,
    "sc": 130,
    "scn": 131,
    "sdh": 132,
    "shi": 133,
    "si": 134,
    "sk": 135,
    "skr": 136,
    "sl": 137,
    "snk": 138,
    "so": 139,
    "sq": 140,
    "sr": 141,
    "ss": 142,
    "st": 143,
    "sv-SE": 144,
    "sw": 145,
    "syr": 146,
    "ta": 147,
    "te": 148,
    "tg": 149,
    "th": 150,
    "ti": 151,
    "tig": 152,
    "tk": 153,
    "tl": 154,
    "tn": 155,
    "tok": 156,
    "tr": 157,
    "ts": 158,
    "tt": 159,
    "tw": 160,
    "ty": 161,
    "uby": 162,
    "udm": 163,
    "ug": 164,
    "uk": 165,
    "ur": 166,
    "uz": 167,
    "ve": 168,
    "vec": 169,
    "vi": 170,
    "vot": 171,
    "xh": 172,
    "yi": 173,
    "yo": 174,
    "yue": 175,
    "zgh": 176,
    "zh-CN": 177,
    "zh-HK": 178,
    "zh-TW": 179,
    "zu": 180
}

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
		languages = []

		for (a, t) in data:
			mels.append(get_log_melspec(a[0], a[1]))
			languages.append(LANGUAGES.get(t['original_data']['language'], -1))

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
			"languages": languages,
		}

		return new_labels, mels, text_lengths, mel_lengths