import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torchaudio

import pytorch_lightning as pl
import webdataset as wds

from typing import Optional

from text.simple_cleaner import text_to_sequence
from text.whisper.normalizers import EnglishTextNormalizer
from text.tokeniser import Tokeniser # from text.whisper.tokenizer import get_tokenizer

import audio as Audio

class MultilingualWebdatasetDataModule(pl.LightningDataModule):
	def __init__(
			self, train_data_dir:str, 
			test_data_dir:str, 
			valid_data_dir:str,
			epochs:Optional[int]=1,
			batch_size:Optional[int]=None,
			num_workers:Optional[int]=0,
			shuffle:Optional[bool]=True,
			resample:Optional[bool]=False,
			audio_backend:Optional[str]=None):
		super().__init__()
		if audio_backend:
			torchaudio.set_audio_backend(audio_backend) # Forching backend to soundfile, due to known bug in torch audio (https://github.com/pytorch/audio/issues/2356)

		self.train_data_dir = train_data_dir
		self.test_data_dir = test_data_dir
		self.valid_data_dir = valid_data_dir

		self.epochs = epochs
		self.shuffle = shuffle
		self.resample = resample
		self.batch_size = batch_size
		self.num_workers = num_workers

		self.pipelines = {
			'train' : self._create_pipeline(self.train_data_dir),
			'test' : self._create_pipeline(self.test_data_dir),
			'valid' : self._create_pipeline(self.valid_data_dir),
		}

		self.cleaner = EnglishTextNormalizer()
		self.tokenizer = Tokeniser() # self.tokenizer = get_tokenizer(True)
		self.stft_fn = Audio.stft.MelSpecPipeline()

	def _create_pipeline(self, data_dir):
		pipeline = []
		if self.resample:
			pipeline.extend([wds.ResampledShards(data_dir)])
		else:
			pipeline.extend([
				wds.SimpleShardList(data_dir),
				wds.detshuffle(),
				wds.split_by_node,
				wds.split_by_worker
				])

		pipeline.extend([wds.tarfile_to_samples()])

		if self.shuffle:
			pipeline.extend([wds.shuffle()])

		pipeline.extend([
			wds.decode(wds.torch_audio),
			wds.to_tuple("flac", "json"),
			wds.batched(self.batch_size),
			wds.map(self.collate_fn)
			])
		return pipeline

	def setup(self, stage:Optional[str] = None):
		if len(self.train_data_dir)>0:
			self.train = wds.DataPipeline(*self.pipelines['train'])
			if self.resample:
				self.train = self.train.with_epoch(self.epochs)
		if len(self.test_data_dir)>0:
			self.test = wds.DataPipeline(*self.pipelines['test'])
			if self.resample:
				self.test = self.test.with_epoch(self.epochs)
		if len(self.valid_data_dir)>0:
			self.valid = wds.DataPipeline(*self.pipelines['valid'])
			if self.resample:
				self.valid = self.valid.with_epoch(self.epochs)

	def train_dataloader(self):
		if self.train:
			return wds.WebLoader(self.train, batch_size=None, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

	def val_dataloader(self):
		if self.valid:
			return wds.WebLoader(self.valid, batch_size=None, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

	def test_dataloader(self):
		if self.test:
			return wds.WebLoader(self.test, batch_size=None, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

	def predict_dataloader(self):
		if self.test:
			return wds.WebLoader(self.test, batch_size=None, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

	# 	return text, mel
	def collate_fn(self, data):
		raw_audios, raw_texts = data

		mels = [self.stft_fn(audio[0][0]).T for audio in raw_audios]
		if isinstance(raw_texts[0]['text'], list):
			texts = [torch.tensor(self.tokenizer.encode(self.cleaner(text['text'][0]))) for text in raw_texts]
		elif isinstance(raw_texts[0]['text'], str):
			texts = [torch.tensor(self.tokenizer.encode(self.cleaner(text['text']))) for text in raw_texts]
		else:
			raise ValueError('Unsupported text type, must be list[str] or str')

		mel_lengths = [mel.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		text_lengths = [text.shape[0] for text in texts]
		text_lengths = torch.tensor(text_lengths)

		texts = pad_sequence(texts).T
		mels = pad_sequence(mels).permute(1,2,0)

		return texts, mels, text_lengths, mel_lengths
	
class WebdatasetDataModule(pl.LightningDataModule):
	def __init__(self, train_data_dir:str, test_data_dir:str, valid_data_dir:str, epochs:int=1, batch_size:int = 32, num_workers:int=0, audio_backend:str=None):
		super().__init__()
		# if not audio_backend:
		# torchaudio.set_audio_backend('soundfile') # Forching backend to soundfile, due to known bug in torch audio (https://github.com/pytorch/audio/issues/2356)

		self.train_data_dir = train_data_dir
		self.test_data_dir = test_data_dir
		self.valid_data_dir = valid_data_dir

		self.epochs = epochs
		self.batch_size = batch_size
		self.num_workers = num_workers

		self.stft_fn = Audio.stft.MelSpecPipeline()

	def setup(self, stage:Optional[str] = None):
		pipeline = [wds.SimpleShardList(self.train_data_dir),
					wds.tarfile_to_samples(),
					wds.detshuffle(),
					wds.split_by_node,
					wds.split_by_worker,
					wds.decode(wds.torch_audio),
					wds.to_tuple("flac", "json"),
					wds.batched(self.batch_size),
					wds.map(self.collate_fn),
					]
		if len(self.train_data_dir)>0:
			self.train = wds.DataPipeline(*pipeline)
		if len(self.test_data_dir)>0:
			self.test = wds.DataPipeline(*pipeline)
		if len(self.valid_data_dir)>0:
			self.valid = wds.DataPipeline(*pipeline)

	def train_dataloader(self):
		if self.train:
			return wds.WebLoader(self.train, batch_size=None, shuffle=False, num_workers=self.num_workers)

	def val_dataloader(self):
		if self.valid:
			return wds.WebLoader(self.valid, batch_size=None, shuffle=False, num_workers=self.num_workers)

	def test_dataloader(self):
		if self.test:
			return wds.WebLoader(self.test, batch_size=None, shuffle=False, num_workers=self.num_workers)

	def predict_dataloader(self):
		if self.test:
			return wds.WebLoader(self.test, batch_size=None, shuffle=False, num_workers=self.num_workers)

	# 	return text, mel
	def collate_fn(self, data):
		raw_audios, raw_texts = data

		mels = [self.stft_fn(audio[0][0]).T for audio in raw_audios]
		if isinstance(raw_texts[0]['text'], list):
			texts = [torch.tensor(text_to_sequence(text['text'][0], ["english_cleaners"])) for text in raw_texts]
		elif isinstance(raw_texts[0]['text'], str):
			texts = [torch.tensor(text_to_sequence(text['text'], ["english_cleaners"])) for text in raw_texts]
		else:
			raise ValueError('Unsupoted text type, must be list[str] or str')

		texts = pad_sequence(texts).T
		mels = pad_sequence(mels).permute(1,2,0)

		return texts, mels
	
if __name__ == '__main__':
	import tqdm
	from utils.get_wds_urls import get_tar_path_s3
	dataset_names = [
		'EmoV_DB', #PASS
	]
	urls = get_tar_path_s3(
		base_s3_path = 's-laion-audio/webdataset_tar/', 
		train_valid_test = ['train', 'test', 'valid'],
		dataset_names = dataset_names,
		# cache_path = '/tmp/url_cache.json',
		# recache = True,
		)
	dataset = MultilingualWebdatasetDataModule(	
									train_data_dir = urls['train'][:1], 
									test_data_dir =urls['test'][:1], 
									valid_data_dir = urls['valid'][:1], 
									batch_size = 64,
									num_workers=6,
									resample=False)

	dataset.setup()

	# print(len(dataset.train))
	# for i, val in enumerate(iter(dataset.train)):
		# print(i)
	# for i in tqdm.tqdm(dataset.train_dataloader()):
	# 	print(i[0].shape, i[1].shape)
	# 	break
	# for i in tqdm.tqdm(dataset.val_dataloader()):
	# 	pass
	# for i in tqdm.tqdm(dataset.test_dataloader()):
	# 	pass