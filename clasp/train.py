from argparse import ArgumentParser

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

import logging
pl_logger = logging.getLogger('pytorch_lightning')

from clasp import CLASP
from loss import CLAPLoss
from datamodules import MultilingualWebdatasetDataModule
from utils import get_tar_path_s3, Accuracy

class PL_CLASP(pl.LightningModule):
	def __init__(	self, 
					hidden_dim=128, 
					learning_rate=1e-3, 
					learning_rate_patience=10, 
					text_encoder_width=1024,
					text_encoder_embedding=1024,
					text_encoder_layers=1,
					text_encoder_heads=4,
					vocab_size=50373,
					n_mels=80,
					audio_encoder_embedding=1024,
					debug=False,
					):

		super().__init__()
		self.save_hyperparameters()

		self.model = CLASP(self.hparams)
		self.loss_fn = CLAPLoss(cache_labels=True)
		self.acc_fn = Accuracy(cache_labels=True)

	def forward(self, batch):
		texts, mels, text_lengths, mel_lengths  = batch # torch.size([*, 123]), torch.size([*,80,1234])
		return self.model(texts, mels)

	def training_step(self, batch, batch_idx):
		model_out = self(batch)
		loss = self.loss_fn(*model_out)
		
		self.log('text_temp', model_out[2])
		self.log('audio_temp', model_out[3])
		self.log('train_loss', loss, prog_bar=True, sync_dist=True)

		if self.hparams.debug and self.current_epoch!= 0 and self.current_epoch%20 == 0:
			breakpoint()
		return loss

	def validation_step(self, batch, batch_idx):
		loss, acc = self._shared_eval_step(batch, batch_idx)

		metrics = {"val_acc": acc, "val_loss": loss}
		self.log_dict(metrics, prog_bar=True, sync_dist=True)

	def test_step(self, batch, batch_idx):
		loss, acc = self._shared_eval_step(batch, batch_idx)

		metrics = {"test_acc": acc, "test_loss": loss}
		self.log_dict(metrics, sync_dist=True)

	def _shared_eval_step(self, batch, batch_idx):
		model_out = self(batch)

		loss = self.loss_fn(*model_out)
		acc = self.acc_fn(*model_out)

		return loss, acc


	def predict_step(self, batch, batch_idx, dataloader_idx=0):
		return self(batch)

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
		lr_scheduler = {
			'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10),
			'name': 'lr_scheduler',
			'monitor': 'valid_loss',
		}
		return [optimizer], [lr_scheduler]

	@staticmethod
	def add_model_specific_args(parent_parser):
		from text.simple_cleaner.symbols import symbols

		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--hidden_dim', type=int, default=128)
		parser.add_argument('--learning_rate', type=float, default=1e-3)
		parser.add_argument('--learning_rate_patience', type=int, default=20)
		parser.add_argument('--text_encoder_width', type=int, default=1024)
		parser.add_argument('--text_encoder_embedding', type=int, default=1024)
		parser.add_argument('--text_encoder_layers', type=int, default=1)
		parser.add_argument('--text_encoder_heads', type=int, default=4)
		parser.add_argument('--vocab_size', type=int, default=50373)
		parser.add_argument('--debug', type=bool, default=False)

		return parser

def cli_main():
	pl.seed_everything(9876)

	# ------------
	# args
	# ------------
	parser = ArgumentParser()
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--num_workers', default=6, type=int)
	parser.add_argument('--early_stoping_patience', type=int, default=10)
	parser.add_argument('--monitor_lr', type=bool, default=True)
	parser.add_argument('--checkpoint', type=str, default=None)
	parser.add_argument('--name', type=str, default=None)
	parser.add_argument('--predict', type=bool, default=False)

	parser.add_argument('--testing_stuff', type=bool, default=False)

	parser = pl.Trainer.add_argparse_args(parser)
	parser = PL_CLASP.add_model_specific_args(parser)
	args = parser.parse_args()

	# ------------
	# data
	# ------------
	exclude = [
		'epidemic_sound_effects/train/1.tar',
		'Tunebot/train/15.tar',
		'Tunebot/train/8.tar',
		# long text
		'common_voice/train/60.tar',
		'common_voice/train/227.tar',
		'common_voice/train/469.tar',
		'common_voice/train/595.tar',
		'common_voice/train/677.tar',
		'common_voice/train/1292.tar',
		'common_voice/train/1001.tar',
		'common_voice/train/1183.tar',
		'common_voice/test/4.tar',
		'common_voice/test/36.tar',
		'common_voice/test/80.tar',
		'FSD50K/train/0.tar',
		'FSD50K/train/2.tar',
		'FSD50K/train/7.tar',
		'FSD50K/train/24.tar',
		'FSD50K/train/28.tar',
		'FSD50K/train/33.tar',
		'FSD50K/train/36.tar',
		'FSD50K/train/42.tar',
		'FSD50K/train/45.tar',
		'FSD50K/train/52.tar',
		'FSD50K/train/60.tar',
		'FSD50K/valid/0.tar',
		# Long audio files
		'common_voice/train/702.tar',
		'common_voice/train/723.tar',
		'common_voice/train/788.tar',
		'common_voice/train/1079.tar', # long file 33.876s
		'common_voice/train/1301.tar',
		'common_voice/train/1655.tar',
		'common_voice/train/1683.tar',

		# zero length
		'midi50k/train/36.tar',
		'midi50k/test/0.tar',


		# Other

		#unsuported languages
		"mswc/ga-IE/",
		"mswc/fy-NL/",
		"mswc/ia/",
		"mswc/sah/",
		"mswc/rm-sursilv/",
		"mswc/rm-vallader/",
		"mswc/sv-SE/",
		"mswc/cnh/",
	]
	dataset_names = [
		'CMU_Arctic', 'Clotho', 'audiocaps', 'EmoV_DB', 'Knocking_sounds', 'LibriSpeech', 
		'esc50_no_overlap', 'cambridge_dictionary', 'fine_grained_vocal_imitation_set', 
		'VocalSketch', 'ESC50_1','ESC50_2','ESC50_3','ESC50_4','ESC50_5','Urbansound8K', 
		'Tunebot', 'MACS', 'LibriSpeech', 'VGGSound', 'audioset_unbalanced_train_t5', 
		'audioset_eval_t5', 'CREMA-D', 'midi50k', 'common_voice']
	dataset_names = [
		# 'FSD50K', 
		# 'audioset', #
		'mswc',
	]
	
	dataset_names_intersection = set(dataset_names).intersection(exclude)
	if dataset_names_intersection:
		raise Warning(f'Found similary dataset names in datasets and excluded dataset: {dataset_names_intersection}')
	
	pl_logger.info(f"Dataset names: \n{dataset_names}\n")

	if args.overfit_batches:
		urls = {
			'train':['/fsx/knoriy/processed_datasets/clasp_local_data/train/0.tar'], 
			'test':['/fsx/knoriy/processed_datasets/clasp_local_data/train/0.tar'], 
			'valid':['/fsx/knoriy/processed_datasets/clasp_local_data/train/0.tar']
		}
	else:
		urls = get_tar_path_s3(
			base_s3_path		= 's-laion-audio/webdataset_tar/', 
			train_valid_test	= ['train', 'test', 'valid'],
			dataset_names		= dataset_names, 
			exclude				= exclude,
			# cache_path			= './tmp/url_cache.json',
			# use_cache			= True,
			# recache				= True,
			)
		if not urls['valid']:
			urls['valid'] = ['/fsx/knoriy/processed_datasets/clasp_local_data/train/0.tar']
		if not urls['test']:
			urls['test'] = ['/fsx/knoriy/processed_datasets/clasp_local_data/train/0.tar']

	pl_logger.info(f"Urls found: \
		\n\t{len(urls['train'])} train \
		\n\t{len(urls['valid'])} valid \
		\n\t{len(urls['test'])} test"
	)

	assert urls['train'], "Train URLs is empty"
	assert urls['valid'], "Valid URLs is empty"
	assert urls['test'], "Test URLs is empty"

	dataset = MultilingualWebdatasetDataModule(	
					train_data_dir = urls['train'],
					test_data_dir = urls['test'],
					valid_data_dir = urls['valid'],
					batch_size = args.batch_size,
					num_workers = args.num_workers,
					shuffle = False if args.overfit_batches else False,
					resample = False,
					)

	# ------------
	# model
	# ------------
	model = PL_CLASP(args.hidden_dim, args.learning_rate, vocab_size=args.vocab_size)
	if os.path.isfile(str(args.checkpoint)):
		model = model.load_from_checkpoint(str(args.checkpoint))
		pl_logger.info(f"Model state loaded from checkpoint: {args.checkpoint}")

	# ------------
	# Callbacks
	# ------------
	callbacks = [
		# ModelCheckpoint(verbose=True, every_n_train_steps=1000)
		# EarlyStopping(monitor="val_loss", patience=args.early_stoping_patience)
	]

	# ------------
	# Loggers
	# ------------
	logger = None
	if args.logger and not args.fast_dev_run:
		from pytorch_lightning.loggers import WandbLogger
		logger = WandbLogger(name=args.name, save_dir="logs/", project="CLASP")
		if args.monitor_lr:
			callbacks.append(LearningRateMonitor())

	# ------------
	# Other
	# ------------
	strategy = None
	if args.strategy == 'ddp':
		strategy = DDPStrategy(find_unused_parameters=False)
	else:
		strategy = args.strategy
		
	# ------------
	# Get Trainer
	# ------------
	trainer = pl.Trainer.from_argparse_args(args, 
		callbacks=callbacks,
		logger=logger,
		strategy=strategy,
	)
	
	if not args.predict:
		# ------------
		# training
		# ------------
		trainer.fit(model, datamodule=dataset)

		# ------------
		# testing
		# ------------
		if not args.fast_dev_run:
			print('Running test')
			trainer.test(ckpt_path='best', datamodule=dataset)
	else:
		# import matplotlib.pyplot as plt
		model = model.load_from_checkpoint("/fsx/knoriy/code/CLASP/.archive/epoch=33-step=2652.ckpt")
		predictions = trainer.predict(model, dataloaders=dataset)

		# print(len(predictions))
		# for prediction in predictions:
		# 	text_features, audio_features, text_tempeture, audio_tempeture, mlp_text_features, mlp_audio_features = prediction

		# 	logits = text_tempeture * text_features @ mlp_audio_features.T
		# 	audio_similarity = mlp_audio_features @ mlp_audio_features.T
		# 	texts_similarity = mlp_text_features @ mlp_text_features.T
		# 	targets = F.softmax(
		# 		((audio_similarity + texts_similarity) / 2) * text_tempeture, dim=-1
		# 	)

		# 	texts_loss = F.cross_entropy(logits, targets, reduction='mean')
		# 	images_loss = F.cross_entropy(logits.T, targets.T, reduction='mean')

		# 	plt.imsave('_logits.png', logits)
		# 	plt.imsave('_audio_similarity.png', audio_similarity)
		# 	plt.imsave('_texts_similarity.png', texts_similarity)
		# 	plt.imsave('_sub_aud_sim.png', logits - audio_similarity)

		# 	print(texts_loss, images_loss)
		# 	break


if __name__ == '__main__':
	cli_main()