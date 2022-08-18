from itertools import count
from operator import truediv
import os
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchvision.datasets.mnist import MNIST
from torchvision import transforms

import pandas as pd
import pytorch_lightning as pl
import webdataset as wds

from typing import Optional

from text import text_to_sequence, sequence_to_text

class MNISTDataModule(pl.LightningDataModule):
	def __init__(self, data_dir: str = "", batch_size: int = 32):
		super().__init__()
		self.data_dir = data_dir
		self.batch_size = batch_size

	def setup(self, stage:Optional[str] = None):
		dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
		self.mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
		self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])

	def train_dataloader(self):
		return DataLoader(self.mnist_train, batch_size=self.batch_size)

	def val_dataloader(self):
		return DataLoader(self.mnist_val, batch_size=self.batch_size)

	def test_dataloader(self):
		return DataLoader(self.mnist_test, batch_size=self.batch_size)

class LJSpeechDataset(Dataset):
	'''
	Dataset loader
	'''
	def __init__(self, preprocessed_path, filepath:str, text_cleaners:Optional[str]=["english_cleaners"], sep='|'):
		self.preprocessed_path = preprocessed_path
		self.text_cleaners = text_cleaners

		self.df = pd.read_csv(os.path.join(preprocessed_path, filepath), header=None, sep=sep)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		file_name = self.df.iloc[idx][0]
		phone = torch.tensor(text_to_sequence(self.df.iloc[idx][3], self.text_cleaners))

		mel_path = os.path.join( self.preprocessed_path, "mels", f"{file_name}.pt")
		mel = torch.load(mel_path)

		return phone, mel
	
	def collate_fn(self, data):
		# split values into own varable
		texts = [i[0] for i in data]
		mels = [i[1].T for i in data]

		# get original length of elements
		text_lens = [text.shape[0] for text in texts]
		mel_lens = [mel.shape[1] for mel in mels]

		# zero pad
		text = pad_sequence(texts).T
		mels = pad_sequence(mels).permute(1,2,0)

		return text, text_lens, mels, mel_lens

class LJSpeechDataModule(pl.LightningDataModule):
	def __init__(self, data_dir: str = "", batch_size: int = 32):
		super().__init__()
		self.data_dir = data_dir # 'pipe:aws s3 --cli-connect-timeout 0 cp s3://s-laion-audio/webdataset_tar/audioset/{0..10}.tar -'
		self.batch_size = batch_size

	def setup(self, stage:Optional[str] = None):
		dataset = LJSpeechDataset(self.data_dir, 'processed.csv')
		train, val, test = len(dataset)*0.7, len(dataset)*0.2, len(dataset)*0.1 
		# self.train, self.val, self.test = random_split(dataset, [train, val, test])
		self.train = self.val = self.test = dataset

	def train_dataloader(self):
		return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.train.collate_fn)

	def val_dataloader(self):
		return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.val.collate_fn)

	def test_dataloader(self):
		return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.test.collate_fn)

class WebdatasetDataModule(pl.LightningDataModule):
	def __init__(self, train_data_dir:str, test_data_dir:str, valid_data_dir:str, batch_size:int = 32):
		super().__init__()
		self.train_data_dir = train_data_dir
		self.test_data_dir = test_data_dir
		self.valid_data_dir = valid_data_dir

		self.batch_size = batch_size

	def setup(self, stage:Optional[str] = None):
		self.train =  wds.WebDataset(self.train_data_dir, verbose=True).decode(wds.torch_audio)
		self.test =  wds.WebDataset(self.test_data_dir).decode(wds.torch_audio)
		self.valid =  wds.WebDataset(self.valid_data_dir).decode(wds.torch_audio)

	def train_dataloader(self):
		return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_fn)

	def val_dataloader(self):
		return DataLoader(self.valid, batch_size=self.batch_size, collate_fn=self.collate_fn)

	def test_dataloader(self):
		return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn)

	def collate_fn(self, data):
		mel_spec_fn = torchaudio.transforms.MelSpectrogram(48000, n_mels = 80)

		# split values into own varable
		texts = []
		mels = []
		for i in data:
			try:
				mels.append(mel_spec_fn(i['flac'][0][0]).T)
				texts.append(torch.tensor(text_to_sequence(i['json']['text'], ["english_cleaners"])))
			except:
				pass

		# get original length of elements
		# text_lens = [text.shape[0] for text in texts]
		# mel_lens = [mel.shape[1] for mel in mels]

		# zero pad
		texts = pad_sequence(texts).T
		mels = pad_sequence(mels).permute(1,2,0).unsqueeze(1)

		return texts, mels

if __name__ == '__main__':
	dataset = WebdatasetDataModule(	train_data_dir = 'pipe:aws s3 cp s3://s-laion-audio/webdataset_tar/audiocaps/train/{0..89}.tar -', 
									test_data_dir ='pipe:aws s3 cp s3://s-laion-audio/webdataset_tar/audiocaps/test/{0..8}.tar -', 
									valid_data_dir = 'pipe:aws s3 cp s3://s-laion-audio/webdataset_tar/audiocaps/valid/{0..4}.tar -', 
									batch_size = 512)

	dataset.setup()
	_count = 0
	for i in dataset.train_dataloader():
		print(_count)
		for x in i:
			print(x.shape)
		_count +=1
		# break