from argparse import ArgumentParser
from typing import Optional

import os
import signal
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment

from scheduler import CosineAnnealingWithWarmup

import logging
pl_logger = logging.getLogger('pytorch_lightning')

from clasp import CLASP
from loss import CLAPLoss, CLIPLoss
from td_datamodules import MultilingualTorchDataDataModule
from utils import get_s3_paths, get_lists, Accuracy

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
		# self.acc_fn = Accuracy(cache_labels=True)

	def forward(self, texts:Optional[torch.Tensor], mels:Optional[torch.Tensor]):
		return self.model(texts, mels)
	
	def encode_audio(self, mels:Optional[torch.Tensor]):
		return self.model.encode_audio(mels)

	def encode_text(self, text:Optional[torch.Tensor]):
		return self.model.encode_text(text)
	
	def get_temps(self):
		return self.model.text_tempeture.exp(), self.model.audio_tempeture.exp()

	def training_step(self, batch, batch_idx):
		texts, mels, _, _  = batch # torch.size([*, 123]), torch.size([*,80,1234])

		model_out = self(texts, mels)
		loss = self.loss_fn(*model_out)
		
		self.log('text_temp', model_out[2])
		self.log('audio_temp', model_out[3])
		self.log('train_loss', loss, prog_bar=True, sync_dist=True)

		return loss

	# def validation_step(self, batch, batch_idx):
	# 	_, loss, acc = self._shared_eval_step(batch, batch_idx)

	# 	metrics = {"val_acc": acc, "val_loss": loss}
	# 	self.log_dict(metrics, prog_bar=True, sync_dist=True)

	def test_step(self, batch, batch_idx):
		_, loss, acc = self._shared_eval_step(batch, batch_idx)

		metrics = {"test_acc": acc, "test_loss": loss}
		self.log_dict(metrics, sync_dist=True)

	def _shared_eval_step(self, batch, batch_idx):
		texts, mels, _, _  = batch # torch.size([*, 123]), torch.size([*,80,1234])
		model_out = self(texts, mels)

		loss = self.loss_fn(*model_out)
		# acc = self.acc_fn(*model_out)
		acc = torch.tensor(0.0)

		return model_out, loss, acc

	def predict_step(self, batch, batch_idx, dataloader_idx=0):
		model_out, loss, acc = self._shared_eval_step(batch, batch_idx)
		return model_out, loss, acc

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
		lr_scheduler = {
			'scheduler': CosineAnnealingWithWarmup(optimizer=optimizer, T_max=200, warmup_steps=20),
			'name': 'lr_scheduler',
			'monitor': 'valid_loss',
		}
		return [optimizer], [lr_scheduler]

	@staticmethod
	def add_model_specific_args(parent_parser):
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
	torch.set_float32_matmul_precision('medium')

	# ------------
	# args
	# ------------
	parser = ArgumentParser()
	parser.add_argument('--batch_size', default=16, type=int)
	parser.add_argument('--num_workers', default=6, type=int)
	parser.add_argument('--persistent_workers', default=True, type=int)
	parser.add_argument('--early_stoping_patience', type=int, default=10)
	parser.add_argument('--monitor_lr', type=bool, default=True)
	parser.add_argument('--checkpoint', type=str, default=None)
	parser.add_argument('--name', type=str, default=None)
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'predict', 'eval-zeroshot'], help='The mode in which to run the script: train a new model, predict using an existing model, or evaluate the performance of an existing model.')
	parser.add_argument('--dataset_list', type=str, default='/fsx/knoriy/code/CLASP/config/dataset_list.txt')
	parser.add_argument('--exclude_list', type=str, default='/fsx/knoriy/code/CLASP/config/exclude_list.txt')
	parser.add_argument('--zeroshot_templates', type=str, default='/fsx/knoriy/code/CLASP/config/zeroshot_templates.txt')

	parser = pl.Trainer.add_argparse_args(parser)
	parser = PL_CLASP.add_model_specific_args(parser)
	args = parser.parse_args()

	# ------------
	# data
	# ------------
	exclude = get_lists(args.exclude_list)
	dataset_names = get_lists(args.dataset_list)
	
	dataset_names_intersection = set(dataset_names).intersection(exclude)
	if dataset_names_intersection:
		raise Warning(f'Found similary dataset names in datasets and excluded dataset: {dataset_names_intersection}')
	
	pl_logger.info(f"Dataset names: \n{dataset_names}\n")

	urls = get_s3_paths(
		base_s3_path		= 's-laion-audio/webdataset_tar/', 
		train_valid_test	= ['train', 'test', 'valid'],
		dataset_names		= dataset_names, 
		exclude				= exclude,
		cache_path			= f"./tmp/{os.path.basename(args.dataset_list)}.json",
		use_cache			= True
		)
	if args.overfit_batches:
		urls = {
			'train':urls['train'][:1], 
			'test':urls['test'][:1], 
			'valid':urls['valid'][:1]
		}

	pl_logger.info(f"Urls found: \
		\n\t{len(urls['train'])} train \
		\n\t{len(urls['valid'])} valid \
		\n\t{len(urls['test'])} test"
	)

	assert urls['train'], "Train URLs is empty"
	assert urls['valid'], "Valid URLs is empty"
	assert urls['test'], "Test URLs is empty"

	dataset = MultilingualTorchDataDataModule(	
					train_data_dir = urls['train'],
					test_data_dir = urls['test'],
					valid_data_dir = urls['valid'],
					batch_size = args.batch_size,
					num_workers = args.num_workers,
					persistent_workers = args.persistent_workers,
					shuffle = False if args.overfit_batches else True,
					)

	# ------------
	# model
	# ------------
	model = PL_CLASP(args.hidden_dim, args.learning_rate, vocab_size=args.vocab_size)
	if os.path.isfile(str(args.checkpoint)):
		model = model.load_from_checkpoint(str(args.checkpoint))
		pl_logger.info(f"Model state loaded from checkpoint: {args.checkpoint}")
	elif args.checkpoint != None:
		pl_logger.warning(f"Checkpoint not found: {args.checkpoint}")

	# ------------
	# Callbacks
	# ------------
	callbacks = [
		ModelCheckpoint(verbose=True)
		# EarlyStopping(monitor="val_loss", patience=args.early_stoping_patience)
	]

	# ------------
	# Loggers
	# ------------
	logger = None
	if args.logger and not args.fast_dev_run and args.mode == 'train':
		os.makedirs("logs/", exist_ok=True)
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
	
	plugins = None
	plugins = [SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGUSR1)]

	# ------------
	# Get Trainer
	# ------------
	trainer = pl.Trainer.from_argparse_args(args, 
		callbacks=callbacks,
		logger=logger,
		strategy=strategy,
		plugins=plugins,
	)
	
	pl_logger.info(f'{f" Mode: {args.mode} ":*^50}')
	if args.mode == 'train':
		trainer.fit(model, datamodule=dataset, ckpt_path=args.checkpoint)

	if args.mode == 'test' and not args.fast_dev_run:
		trainer.test(ckpt_path='best', datamodule=dataset)

	if args.mode == 'predict':
		predictions = trainer.predict(model, dataloaders=dataset)

		for prediction in predictions:
			model_out, loss, acc = prediction
			print("\n")
			print(f"audio features: {model_out[0].shape}")
			print(f"text features: {model_out[1].shape}")
			print(loss, acc)
			break
	
	if args.mode == 'eval-zeroshot':
		from eval.zeroshot import zeroshot_eval

		dataset.setup()
		templates = get_lists(args.zeroshot_templates)
		classes = ["hello world", "how are you?", "some random thing", "it's a beautiful day", "i love you", "goodbye", "i hate you", "today is not the day"]

		acc1, acc5 = zeroshot_eval(model, classes, templates, dataset.val_dataloader())
		pl_logger.info(f"acc1: {acc1:.3f}, acc5: {acc5:.3f}")

	pl_logger.info(f'{" The END ":*^50}')

if __name__ == '__main__':
	cli_main()