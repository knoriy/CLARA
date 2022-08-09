from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms

import pytorch_lightning as pl


from typing import Optional

from clap import CLAP

class LitClassifier(pl.LightningModule):
	def __init__(	self, 
					hidden_dim=128, 
					learning_rate=1e-3, 
					text_encoder_width=1024,
					text_encoder_embedding=1024,
					text_encoder_layers=1,
					text_encoder_heads=4):

		super().__init__()
		self.save_hyperparameters()

		self.model = CLAP(self.hparams)

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = F.cross_entropy(y_hat, y)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = F.cross_entropy(y_hat, y)
		self.log('valid_loss', loss)

	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = F.cross_entropy(y_hat, y)
		self.log('test_loss', loss)

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

	@staticmethod
	def add_model_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--hidden_dim', type=int, default=1280)
		parser.add_argument('--learning_rate', type=float, default=0.0001)

		parser.add_argument('--text_encoder_width', type=int, default=1024)
		parser.add_argument('--text_encoder_embedding', type=int, default=1024)
		parser.add_argument('--text_encoder_layers', type=int, default=1)
		parser.add_argument('--text_encoder_heads', type=int, default=4)

		return parser

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

def cli_main():
	pl.seed_everything(1234)

	# ------------
	# args
	# ------------
	parser = ArgumentParser()
	parser.add_argument('--batch_size', default=32, type=int)
	parser.add_argument('--num_workers', default=1, type=int)

	parser = pl.Trainer.add_argparse_args(parser)
	parser = LitClassifier.add_model_specific_args(parser)
	args = parser.parse_args()

	# ------------
	# data
	# ------------
	mnist = MNISTDataModule('')

	# ------------
	# model
	# ------------
	model = LitClassifier(args.hidden_dim, args.learning_rate)

	# ------------
	# training
	# ------------
	trainer = pl.Trainer.from_argparse_args(args)
	trainer.fit(model, datamodule=mnist)

	# ------------
	# testing
	# ------------
	trainer.test(datamodule=mnist)


if __name__ == '__main__':
	cli_main()
