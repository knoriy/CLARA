from argparse import ArgumentParser

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from clasp import CLASP
from loss import CLAPLoss
from datamodules import WebdatasetDataModule, MultilingualWebdatasetDataModule
from utils.get_wds_urls import get_tar_path_s3

class PL_CLASP(pl.LightningModule):
	def __init__(	self, 
					hidden_dim=128, 
					learning_rate=1e-3, 
					text_encoder_width=1024,
					text_encoder_embedding=1024,
					text_encoder_layers=1,
					text_encoder_heads=4,
					vocab_size=50257,
					):

		super().__init__()
		self.save_hyperparameters()

		self.model = CLASP(self.hparams)
		self.loss_fn = CLAPLoss(cache_labels=True)

	def forward(self, batch):
		texts, mels = batch # torch.size([61, 123]), torch.size([62,80,1234])
		# texts, mels = texts.squeeze(0), mels.unsqueeze(1) # torch.size([64, 100]), torch.size([64,1,80,100])
		return self.model(texts, mels)
	
	def training_step(self, batch, batch_idx):
		text_features, audio_features, tempeture = self(batch)
		loss = self.loss_fn(text_features, audio_features, tempeture)

		self.log('train_loss', loss, sync_dist=True)
		return {"loss": loss}

	def validation_step(self, batch, batch_idx):
		text_features, audio_features, tempeture = self(batch)
		loss = self.loss_fn(text_features, audio_features, tempeture)

		self.log('valid_loss', loss, sync_dist=True, prog_bar=True)

	def test_step(self, batch, batch_idx):
		text_features, audio_features, tempeture = self(batch)
		loss = self.loss_fn(text_features, audio_features, tempeture)

		self.log('test_loss', loss, sync_dist=True, prog_bar=True)

	def predict_step(self, batch, batch_idx, dataloader_idx=0):
		return self(batch)

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
		# lr_scheduler = {
		# 	'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer),
		# 	'name': 'lr_scheduler',
		# 	'monitor': 'valid_loss',
		# }
		return optimizer

	@staticmethod
	def add_model_specific_args(parent_parser):
		from text.simple_cleaner.symbols import symbols

		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--hidden_dim', type=int, default=128)
		parser.add_argument('--learning_rate', type=float, default=1e-3)
		parser.add_argument('--text_encoder_width', type=int, default=1024)
		parser.add_argument('--text_encoder_embedding', type=int, default=1024)
		parser.add_argument('--text_encoder_layers', type=int, default=1)
		parser.add_argument('--text_encoder_heads', type=int, default=4)
		parser.add_argument('--vocab_size', type=int, default=50257)# len(symbols))



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
	parser.add_argument('--testing_stuff', type=bool, default=False)



	parser = pl.Trainer.add_argparse_args(parser)
	parser = PL_CLASP.add_model_specific_args(parser)
	args = parser.parse_args()

	# ------------
	# data
	# ------------
	dataset_names = [
		# '130000_MIDI_SONGS', #PASS
		# 'CREMA-D', #PASS
		# 'Clotho', #PASS
		# 'CoVoST_2',#PASS
		'EmoV_DB', #PASS
		# 'FSD50K', #PASS
		# 'Urbansound8K', #PASS
		# 'audiocaps', #PASS
		# 'audioset', #PASS
		# 'audiostock', #PASS
		# 'cambridge_dictionary', #PASS
		# 'esc50', #PASS
		# 'free_to_use_sounds', #PASS
		# 'freesound', #PASS
		# 'midi50k', #PASS
		# 'paramount_motion', #PASS
		# 'sonniss_game_effects', #PASS
		# 'wesoundeffects', #PASS
		# 'FMA_updated', #FAIL
		# 'LJSpeech', #FAIL
		# 'VocalSketch', #FAIL
		# 'YT_dataset', #FAIL
		# 'clotho_mixed', #FAIL
		# 'ravdess', #FAIL
		# # 'tmp_eval',
		# 'BBCSoundEffects', #FAIL
	]
	
	urls = get_tar_path_s3(
		base_s3_path		= 's-laion-audio/webdataset_tar/', 
		train_valid_test	= ['train', 'test', 'valid'],
		dataset_names		= dataset_names, 
		# cache_path			= '/tmp/url_cache.json',
		# recache				= True,
		)

	urls = {
		'train':['/fsx/knoriy/processed_datasets/clasp_local_data/train/0.tar'], 
		'test':['/fsx/knoriy/processed_datasets/clasp_local_data/test/0.tar'], 
		'valid':['/fsx/knoriy/processed_datasets/clasp_local_data/valid/0.tar']
	}

	dataset = MultilingualWebdatasetDataModule(	
					train_data_dir = urls['train'],
					test_data_dir = urls['test'],
					valid_data_dir = urls['valid'],
					epochs = args.max_epochs,
					batch_size = args.batch_size,
					num_workers = args.num_workers)

	# ------------
	# model
	# ------------
	model = PL_CLASP(args.hidden_dim, args.learning_rate)

	if not args.testing_stuff:
		# ------------
		# Callbacks
		# ------------
		checkpoint_callback = ModelCheckpoint(monitor="valid_loss")
		early_stopping_callback = EarlyStopping(monitor="valid_loss", patience=args.early_stoping_patience)
		lr_monitor = LearningRateMonitor()

		# ------------
		# training
		# ------------
		trainer = pl.Trainer.from_argparse_args(args, 
			callbacks=[
				# checkpoint_callback,
				# early_stopping_callback, 
				# lr_monitor,
				],
		)
		
		trainer.fit(model, datamodule=dataset)

		# ------------
		# testing
		# ------------
		trainer.test(ckpt_path='best', datamodule=dataset)
	else:
		test_inference(model, dataset)

def test_inference(model:torch.nn.Module, dataset:MultilingualWebdatasetDataModule):
	dataset.setup()

	with torch.no_grad():
		model = model.load_from_checkpoint("/fsx/knoriy/code/CLASP/lightning_logs/version_8344/checkpoints/epoch=99-step=100.ckpt")
		model.eval()
		texts, mels = next(iter(dataset.test_dataloader()))
		texts, mels = texts.squeeze(0), mels
		text_features, audio_features, _ = model((texts, mels))

		text_features, audio_features = text_features.cpu().detach(), audio_features.cpu().detach()

		audio_features /= audio_features.norm(dim=-1, keepdim=True)
		text_features /= text_features.norm(dim=-1, keepdim=True)
		text_probs = (audio_features @ text_features.T).softmax(dim=-1)
		import matplotlib.pyplot as plt
		plt.imsave('test.png', text_probs)
	# texts, mels = next(iter(dataset.test_dataloader()))
	# print(dataset.tokenizer.decode(texts[0]))
	# import librosa
	# import soundfile as sf
	# import matplotlib.pyplot as plt
	# wav = librosa.feature.inverse.mel_to_audio(mels[9].cpu().detach().numpy())
	# sf.write('file_trim_5s.wav', wav, 48000)
	# plt.imsave('mels9.jpeg', mels[0].cpu().detach().numpy())




if __name__ == '__main__':
	cli_main()