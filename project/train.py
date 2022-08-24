from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from clap import CLAP
from loss import CLAPLoss
from datamodules import WebdatasetDataModule
from utils.get_wds_urls import get_tar_path_s3

class PL_CLASP(pl.LightningModule):
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
		self.loss_fn = CLAPLoss(cache_labels=True)

	def forward(self, batch):
		texts, mels = batch
		return self.model(texts, mels)

	def training_step(self, batch, batch_idx):
		text_features, audio_features, tempeture = self(batch)
		loss = self.loss_fn(text_features, audio_features, tempeture)
		self.log('train_loss', loss, sync_dist=True)
		return {"loss": loss}

	def validation_step(self, batch, batch_idx):
		text_features, audio_features, tempeture = self(batch)
		loss = self.loss_fn(text_features, audio_features, tempeture)
		self.log('valid_loss', loss, sync_dist=True)

	def test_step(self, batch, batch_idx):
		text_features, audio_features, tempeture = self(batch)
		loss = self.loss_fn(text_features, audio_features, tempeture)
		self.log('test_loss', loss, sync_dist=True)

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

	@staticmethod
	def add_model_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--hidden_dim', type=int, default=128)
		parser.add_argument('--learning_rate', type=float, default=0.0001)

		parser.add_argument('--text_encoder_width', type=int, default=1024)
		parser.add_argument('--text_encoder_embedding', type=int, default=1024)
		parser.add_argument('--text_encoder_layers', type=int, default=1)
		parser.add_argument('--text_encoder_heads', type=int, default=4)

		return parser

def cli_main():
	pl.seed_everything(1234)

	# ------------
	# args
	# ------------
	parser = ArgumentParser()
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--num_workers', default=6, type=int)

	parser = pl.Trainer.add_argparse_args(parser)
	parser = PL_CLASP.add_model_specific_args(parser)
	args = parser.parse_args()

	# ------------
	# data
	# ------------
	urls = get_tar_path_s3(
		's-laion-audio/webdataset_tar/', 
		['train', 'test', 'valid'],
		['audiocaps'], 
		cache_path='./url_cache.json',
		recache=True,
		)

	dataset = WebdatasetDataModule(	train_data_dir = urls['train'],
									test_data_dir = urls['test'],
									valid_data_dir = urls['valid'],
									batch_size = args.batch_size,
									num_workers = args.num_workers)
	# ------------
	# model
	# ------------
	model = PL_CLASP(args.hidden_dim, args.learning_rate)

	# ------------
	# training
	# ------------
	checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="valid_loss")
	early_stopping_callback = EarlyStopping(monitor="valid_loss")
	lr_monitor = LearningRateMonitor(logging_interval='step')

	trainer = pl.Trainer.from_argparse_args(args, 
		callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
	)
	
	trainer.fit(model, datamodule=dataset)
	print(checkpoint_callback.best_model_path)

	# ------------
	# testing
	# ------------
	trainer.test(ckpt_path='best', datamodule=dataset)


if __name__ == '__main__':
	cli_main()