import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
import webdataset as wds

from typing import Optional

from text import text_to_sequence
import audio as Audio


class WebdatasetDataModule(pl.LightningDataModule):
	def __init__(self, train_data_dir:str, test_data_dir:str, valid_data_dir:str, epochs:int=1, batch_size:int = 32, num_workers:int=0):
		super().__init__()
		self.train_data_dir = train_data_dir
		self.test_data_dir = test_data_dir
		self.valid_data_dir = valid_data_dir

		self.epochs = epochs
		self.batch_size = batch_size
		self.num_workers = num_workers

		self.stft_fn =Audio.stft.TacotronSTFT(
			filter_length=1024,
			hop_length=256,
			win_length=1024,
			n_mel_channels=80,
			sampling_rate=48000,
			mel_fmin=0,
			mel_fmax=8000,
		)

	def setup(self, stage:Optional[str] = None):
		self.train =  wds.WebDataset(self.train_data_dir, resampled=True).decode(wds.torch_audio).to_tuple("flac", "json").batched(self.batch_size).map(self.collate_fn)
		self.test =  wds.WebDataset(self.test_data_dir, resampled=True).decode(wds.torch_audio).to_tuple("flac", "json").batched(self.batch_size).map(self.collate_fn)
		self.valid =  wds.WebDataset(self.valid_data_dir, resampled=True).decode(wds.torch_audio).to_tuple("flac", "json").batched(self.batch_size).map(self.collate_fn)

		# self.train = wds.DataPipeline(
		# 	wds.SimpleShardList(self.train_data_dir),
		# 	# wds.split_by_worker,
		# 	wds.tarfile_to_samples(),
		# 	wds.shuffle(100),
		# 	wds.decode(wds.torch_audio),
		# 	wds.to_tuple("flac", "json"),
		# 	wds.map(self.collate_fn),
		# 	wds.batched(16),
		# )
		# self.test = wds.DataPipeline(
		# 	wds.SimpleShardList(self.test_data_dir),
		# 	# wds.split_by_worker,
		# 	wds.tarfile_to_samples(),
		# 	wds.shuffle(100),
		# 	wds.decode(wds.torch_audio),
		# 	wds.to_tuple("flac", "json"),
		# 	wds.map(self.collate_fn),
		# 	wds.batched(16),
		# )
		# self.valid = wds.DataPipeline(
		# 	wds.SimpleShardList(self.valid_data_dir),
		# 	# wds.split_by_worker,
		# 	wds.tarfile_to_samples(),
		# 	wds.shuffle(100),
		# 	wds.decode(wds.torch_audio),
		# 	wds.to_tuple("flac", "json"),
		# 	wds.map(self.collate_fn),
		# 	wds.batched(16),
		# )

	def train_dataloader(self):
		return wds.WebLoader(self.train, num_workers=self.num_workers)

	def val_dataloader(self):
		return wds.WebLoader(self.valid, num_workers=self.num_workers)

	def test_dataloader(self):
		return wds.WebLoader(self.test, num_workers=self.num_workers)

	# def collate_fn(self, data):
	# 	raw_audios, raw_texts = data
	# 	# split values into own varable
	# 	mel, _  = Audio.tools.get_mel_from_wav(raw_audios[0][0].numpy(), self.stft_fn)
	# 	mel = torch.tensor(mel).T
	# 	text = torch.tensor(text_to_sequence(raw_texts['text'][0], ["english_cleaners"]))

	# 	return text, mel
	def collate_fn(self, data):
		raw_audios, raw_texts = data
		# # split values into own varable
		mels = [Audio.tools.get_mel_from_wav(audio[0][0].numpy(), self.stft_fn)[0] for audio in raw_audios]
		mels = [torch.tensor(mel).T for mel in mels]
		texts = [torch.tensor(text_to_sequence(text['text'][0], ["english_cleaners"])) for text in raw_texts]

		texts = pad_sequence(texts).T
		mels = pad_sequence(mels).permute(1,2,0)

		return texts, mels
	
	def zero_pad(self, batch):
		texts = [data[0] for data in batch]
		mels = [data[1] for data in batch]

		# zero pad
		texts = pad_sequence(texts).T
		mels = pad_sequence(mels).permute(1,2,0)

		return texts, mels

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from utils.get_wds_urls import get_tar_path_s3

	urls = get_tar_path_s3(
		's-laion-audio/webdataset_tar/', 
		['train', 'test', 'valid'],
		# ['EmoV_DB'], 
		# cache_path='./url_cache.json',
		# recache=True,
		)
	dataset = WebdatasetDataModule(	train_data_dir = urls['train'], 
									test_data_dir =urls['test'], 
									valid_data_dir = urls['valid'], 
									batch_size = 1,
									num_workers=0)

	dataset.setup()

	for i in dataset.train_dataloader():
		for j in i[2]:
			print(j[1]['text'])
	# 	break
	# for i in dataset.test_dataloader():
	# 	print(i[1].shape)
	# for i in dataset.val_dataloader():
	# 	print(i[1].shape)