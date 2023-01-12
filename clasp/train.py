from argparse import ArgumentParser

import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy, DDPFullyShardedNativeStrategy

import logging
pl_logger = logging.getLogger('pytorch_lightning')

from clasp import CLASP
from loss import CLAPLoss, CLIPLoss
from datamodules import WebdatasetDataModule, MultilingualWebdatasetDataModule
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
					vocab_size=50362,
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
		parser.add_argument('--vocab_size', type=int, default=50257)# len(symbols))
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

	parser.add_argument('--testing_stuff', type=bool, default=False)

	parser = pl.Trainer.add_argparse_args(parser)
	parser = PL_CLASP.add_model_specific_args(parser)
	args = parser.parse_args()

	# ------------
	# data
	# ------------
	dataset_names = [
		# '', # For Full dataset list
		# '130000_MIDI_SONGS',
		# 'Audiostock_music',
		# 'BBCSoundEffects',
		'CMU_Arctic',
		# 'CREMA-D', fail 'list' object has no attribute 'lower'
		# 'Cambridge_mt', Fail CUDA out of memory
		# 'Clotho',
		'CoVoST_2',
		'ESC50_1',
		'ESC50_2',
		'ESC50_3',
		'ESC50_4',
		'ESC50_5',
		'EmoV_DB',
		# 'Europarl-st', Fail
		# 'FMA',
		# 'FMA_updated', Fail cannot reshape tensor of 0 elements into shape [-1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
		# 'FSD50K', Fail CUDA out of memory
		# 'Genius',Fail CUDA out of memory
		# 'Jamendo',Fail CUDA out of memory
		'Knocking_sounds',
		# 'LJSpeech', Fail Failed to load audio from /tmp/tmpiqgs3s46/file.flac
		'LibriSpeech',
		'MACS',
		# 'MUSDB18-HQ', CUDA out of memory
		'Tunebot',
		'Urbansound8K',
		# 'VGGSound', Fail cannot reshape tensor of 0 elements into shape [-1, 0]
		'VocalSketch',
		# 'WavText5K', Fail 'list' object has no attribute 'lower'
		# 'YT_dataset', cannot reshape tensor of 0 elements into shape [-1, 0]
		# 'ZAPSPLAT', CUDA out of memory.
		'audiocaps',
		# 'audioset', Fail cannot reshape tensor of 0 elements into shape [-1, 0]
		# 'audioset_balanced_train_t5',
		# 'audioset_eval_t5',
		# 'audioset_strong',
		# 'audioset_t5',
		# 'audioset_unbalanced_train_t5',
		# 'audiostock', Cuda out of memory
		'cambridge_dictionary',
		# 'clotho_mixed',
		# 'common_voice', CUDA out of memory. after running half the dataset
		# 'common_voice_11_0', Unknown Languages
		# 'epidemic_sound_effects', Fail, tensor of 0 elements into shape [-1, 0]
		# 'epidemic_sound_effects_t5',
		# 'esc50',
		'esc50_no_overlap',
		'fine_grained_vocal_imitation_set',
		# 'free_to_use_sounds', CUDA out of memory
		# 'freesound',
		# 'freesound_no_overlap',
		# 'freesound_no_overlap_noesc50', Padding size should be less than the corresponding input dimension, but got: padding (512, 512) at dimension 2 of input [1, 1, 89]
		# 'fsd50k_200_class_label',
		# 'fsd50k_class_label',
		# 'juno', # A little slow during training and CUDA out of memory
		# 'librispeech_asr',
		# 'midi50k', cannot reshape tensor of 0 elements into shape [-1, 0] because the unspecified dimension size -1
		# 'million_song_dataset', No train test 
		# 'mswc', Fail Language
		# 'musicnet', Does not conform to audiodataset format
		# 'paramount_motion', CUDA out of memory
		# 'ravdess', Failed to load audio from /tmp/tmppagd9nbx/file.flac, cannot reshape tensor of 0 elements into shape [-1, 0]
		# 'slakh', Fail cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.
		# 'sonniss_game_effects',CUDA out of memory
		# 'tmp_eval',
		# 'urbansound8k_class_label',
		# 'usd8k_no_overlap',
		# 'wesoundeffects',	CUDA out of memory
	]
	exclude = [
		'epidemic_sound_effects/train/1.tar',
		'Tunebot/train/15.tar',
		'Tunebot/train/8.tar',
		# long text
		'common_voice/train/60.tar',
		'common_voice/train/227.tar',
		'common_voice/train/1001.tar',
		'common_voice/train/1183.tar',
		# Long audio files
		'common_voice/train/702.tar',
		'common_voice/train/723.tar',
		'common_voice/train/788.tar',
		'common_voice/train/1079.tar', # long file 33.876s
		'common_voice/train/1301.tar',
		'common_voice/train/1655.tar',
		'common_voice/train/1683.tar',
	]
	# Tested and working datastes
	dataset_names = [
		'CMU_Arctic', 'Clotho', 'audiocaps', 'EmoV_DB', 'Knocking_sounds', 'LibriSpeech', 'esc50_no_overlap', 'cambridge_dictionary', 
		'fine_grained_vocal_imitation_set', 'VocalSketch', 'ESC50_1','ESC50_2','ESC50_3','ESC50_4','ESC50_5','Urbansound8K', 'Tunebot', 
		'MACS', 'LibriSpeech', 'VGGSound', 'audioset_unbalanced_train_t5', 'audioset_eval_t5']

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
			cache_path			= './tmp/url_cache.json',
			use_cache			= True,
			# recache				= True,
			)

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
	model = PL_CLASP(args.hidden_dim, args.learning_rate)
	if os.path.isfile(str(args.checkpoint)):
		model = model.load_from_checkpoint(str(args.checkpoint))
		pl_logger.info(f"Model state loaded from checkpoint: {args.checkpoint}")

	# ------------
	# Callbacks
	# ------------
	callbacks = [
		ModelCheckpoint(verbose=True, every_n_train_steps=1000),
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
	
	if not args.testing_stuff:
		# ------------
		# training
		# ------------
		trainer.fit(model, datamodule=dataset)

		# ------------
		# testing
		# ------------
		if not args.fast_dev_run:
			# trainer.test(ckpt_path='best', datamodule=dataset)
			pass
	else:
		# import matplotlib.pyplot as plt
		# model = model.load_from_checkpoint("/fsx/knoriy/code/CLASP/.archive/epoch=33-step=2652.ckpt")
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