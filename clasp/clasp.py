import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import numpy as np

from typing import Tuple, Union, Callable, Optional, Literal
from encoders.text_encoders import SimpleTransformer 
from encoders.audio_encoders import *
from encoders.modules import PositionalEncoding, LayerNorm, MLPLayers
from loss import CLAPLoss, CLIPLoss

from scheduler import CosineAnnealingWarmupRestarts
from utils.accuracy import Accuracy, accuracy
from einops import rearrange



class CLASP(nn.Module):
	'''
	Contrastive Language-Speech Pre-training 
	'''
	def __init__(self, hparm, text_encoder:Optional[nn.Module]=None, audio_encoder:Optional[nn.Module]=None) -> None:
		super().__init__()
		self.hparm = hparm

		self.text_encoder = text_encoder
		self.audio_encoder = audio_encoder
		
		if self.text_encoder == None:
			self.text_encoder = SimpleTransformer(
				in_channels = self.hparm.text_encoder_embedding,
				out_channels = self.hparm.text_encoder_out_channels,
				num_layers = self.hparm.text_encoder_layers,
				nhead = self.hparm.text_encoder_heads,
				dropout = self.hparm.text_encoder_seq_dropout_prob,
				batch_first = True,
			)
			# self.text_encoder = PerceiverIOEncoder(
			# 	num_layers=self.hparm.text_encoder_layers, 
			# 	dim=self.hparm.text_encoder_embedding, 
			# 	num_latents=self.hparm.text_encoder_out_channels,
			# 	seq_dropout_prob=self.hparm.text_encoder_seq_dropout_prob,
			# 	)

		if self.audio_encoder == None:
			# self.audio_encoder = resnet18(1024)
			# self.audio_encoder = ResNeXt(5,12,1024, 2, 4)
			# self.audio_encoder = WhisperAudioEncoder(80, 1024, 1, 1)
			self.audio_encoder = PerceiverIOEncoder(
				num_layers = self.hparm.audio_encoder_layers,
				dim = self.hparm.audio_encoder_embedding,
				num_latents = self.hparm.audio_encoder_num_latents,
				latent_dim = self.hparm.audio_encoder_latent_dim,
				cross_heads = self.hparm.audio_encoder_cross_heads,
				latent_heads = self.hparm.audio_encoder_latent_heads,
				cross_dim_head = self.hparm.audio_encoder_cross_dim_head,
				latent_dim_head = self.hparm.audio_encoder_latent_dim_head,
				weight_tie_layers = self.hparm.audio_encoder_weight_tie_layers,
				seq_dropout_prob = self.hparm.audio_encoder_seq_dropout_prob
			)

		# ------------
		# Text Layers
		# ------------
		self.text_embedding = nn.Embedding(self.hparm.vocab_size, self.hparm.text_encoder_embedding)
		self.text_positional_embedding = nn.Embedding(self.hparm.text_encoder_pos_embedding_size, self.hparm.text_encoder_embedding)
		self.text_layer_norm = LayerNorm(self.hparm.text_encoder_out_channels)
		self.text_fc1 = nn.Linear(self.hparm.text_encoder_out_channels, self.hparm.text_encoder_project)
		self.text_transform = MLPLayers(
			units=[
				self.hparm.text_encoder_project,
				self.hparm.output_dim,
				], 
			dropout=self.hparm.text_encoder_project_dropout_prob,
		)

		# ------------
		# Audio Layers
		# ------------
		self.conv1 = nn.Conv1d(
			self.hparm.n_mels, 
			self.hparm.audio_encoder_embedding, 
			kernel_size=3, 
			padding=1)
		self.conv2 = nn.Conv1d(
			self.hparm.audio_encoder_embedding, 
			self.hparm.audio_encoder_embedding, 
			kernel_size=3, 
			stride=2, 
			padding=1)
		self.audio_positional_embedding = nn.Embedding(self.hparm.audio_encoder_pos_embedding_size, self.hparm.audio_encoder_embedding)
		self.audio_layer_norm = LayerNorm(self.hparm.audio_encoder_latent_dim)
		self.audio_fc1 = nn.Linear(self.hparm.audio_encoder_latent_dim, self.hparm.audio_encoder_project)
		self.audio_transform = MLPLayers(
			units=[
				self.hparm.audio_encoder_project,
				self.hparm.output_dim,
				], 
			dropout=self.hparm.audio_encoder_project_dropout_prob,
		)

		# ------------
		# Other
		# ------------
		self.audio_tempeture = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
		self.text_tempeture = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

	def encode_text(self, text:torch.Tensor):
		n, device = text.shape[1], text.device
		x = self.text_embedding(text)
		pos_emb = self.text_positional_embedding(torch.arange(n, device = device))
		pos_emb = rearrange(pos_emb, 'n d -> () n d')
		x = x + pos_emb
		# x = x.permute(0,2,1) # (batch, seq, dim) -> (batch, dim, seq)
		x = self.text_encoder(x)
		# x = x.permute(0,2,1) # (batch, dim, seq) -> (batch, seq, dim)
		x = self.text_layer_norm(x)

		x1 = torch.mean(x, 1)
		x2, _ = torch.max(x, 1)
		x = x1 + x2

		x = F.leaky_relu(self.text_fc1(x))

		return x

	def encode_audio(self, audio:torch.Tensor):
		x = F.gelu(self.conv1(audio))
		x = F.gelu(self.conv2(x))
		x = x.permute(0, 2, 1)
		pos_emb = self.audio_positional_embedding(torch.arange(x.shape[1], device = audio.device))
		pos_emb = rearrange(pos_emb, 'n d -> () n d')
		x = x + pos_emb
		x = x.permute(0, 2, 1)

		x = self.audio_encoder(x)
		x = self.audio_layer_norm(x)

		x1 = torch.mean(x, dim=2)
		x2, _ = torch.max(x, dim=2)
		x = x1 + x2

		x = F.leaky_relu(self.audio_fc1(x))

		return x

	def forward(self, text:torch.Tensor, audio:torch.Tensor):
		text_features = self.encode_text(text)
		audio_features = self.encode_audio(audio)

		# Projection
		text_features = self.text_transform(text_features)
		audio_features = self.audio_transform(audio_features)

		text_features = F.normalize(text_features, dim=-1)
		audio_features = F.normalize(audio_features, dim=-1)

		return text_features, audio_features, self.text_tempeture.exp(), self.audio_tempeture.exp()
	
class PLCLASP(pl.LightningModule):
	def __init__(	self, 
					output_dim:int=512, 
					text_encoder_embedding:int=1024,
					text_encoder_pos_embedding_size:int=1024,
					text_encoder_width:int=1024,
					text_encoder_layers:int=1,
					text_encoder_heads:int=4,
					text_encoder_out_channels:int=512,
					text_encoder_project:int=512,
					text_encoder_project_dropout_prob:float=0.1,
					text_encoder_seq_dropout_prob:float=0.5,
					vocab_size:int=50373,

					n_mels:int=80,
					audio_encoder_layers:int=5,
					audio_encoder_embedding:int=512,
					audio_encoder_pos_embedding_size:int=4096,
					audio_encoder_num_latents:int=512,
					audio_encoder_latent_dim:int=512,
					audio_encoder_project:int=512,
					audio_encoder_project_dropout_prob:float=0.1,
					audio_encoder_cross_heads:int=1,
					audio_encoder_latent_heads:int=8,
					audio_encoder_cross_dim_head:int=64,
					audio_encoder_latent_dim_head:int=64,
					audio_encoder_weight_tie_layers:bool=False,
					audio_encoder_seq_dropout_prob:float=0.5,

					learning_rate:float=1e-3, 
					learning_rate_patience:int=10, 
					LR_sheduler_T_max:int=40,
					LR_sheduler_warmup_steps:int=5,
					LR_sheduler_min_lr:float=0.0,
					LR_sheduler_decay:float=1.0,
					lr_interval:Literal["epoch","step"]='epoch',
					):

		super().__init__()
		self.save_hyperparameters()

		self.model = CLASP(self.hparams)
		self.loss_fn = CLAPLoss(cache_labels=True)
		self.acc_fn = Accuracy(cache_labels=True)

	def forward(self, texts:Optional[torch.Tensor], mels:Optional[torch.Tensor]):
		return self.model(texts, mels)
	
	def encode_audio(self, mels:torch.Tensor):
		return self.model.encode_audio(mels)

	def encode_text(self, text:torch.Tensor):
		return self.model.encode_text(text)
	
	def get_temps(self):
		return self.model.text_tempeture.exp(), self.model.audio_tempeture.exp()

	def training_step(self, batch, batch_idx):
		model_out, loss = self._shared_eval_step(batch, batch_idx)
		
		self.log('text_temp', model_out[2], sync_dist=True)
		self.log('audio_temp', model_out[3], sync_dist=True)
		self.log('train_loss', loss, prog_bar=True, sync_dist=True)

		return loss

	def validation_step(self, batch, batch_idx):
		_, loss, acc = self._shared_eval_step(batch, batch_idx)

		metrics = {"val_acc": acc, "val_loss": loss}
		self.log_dict(metrics, prog_bar=True, sync_dist=True)
		return metrics

	def test_step(self, batch, batch_idx):
		_, loss, acc = self._shared_eval_step(batch, batch_idx)

		metrics = {"test_acc": acc, "test_loss": loss}
		self.log_dict(metrics)

	def _shared_eval_step(self, batch, batch_idx):
		texts, mels, _, _  = batch # torch.size([*, 123]), torch.size([*,80,1234])
		model_out = self(texts, mels)

		loss = self.loss_fn(*model_out)

		if self.training:
			return model_out, loss

		acc = self.acc_fn(*model_out)[0] / mels.size(0)
		return model_out, loss, acc

	def predict_step(self, batch, batch_idx, dataloader_idx=0):
		model_out, loss, acc = self._shared_eval_step(batch, batch_idx)
		return model_out, loss, acc

	def configure_optimizers(self):
		return get_optimiser(self)

class LinearProbeCLASP(pl.LightningModule):
	def __init__(self, 
		in_features:int, 
		num_classes:int, 
		task:str,
		clasp_checkpoint_path:str, 
		clasp_map_location:str="cuda",
		dropout:float=0.1,
		learning_rate:float=1e-3, 
		learning_rate_patience:int=10, 
		LR_sheduler_T_max:int=40,
		LR_sheduler_warmup_steps:int=5,
		LR_sheduler_min_lr:float=0.0,
		LR_sheduler_decay:float=1.0,
		lr_interval:Literal["epoch","step"]='epoch',
		*args, **kwargs
	) -> None:
		
		super().__init__(*args, **kwargs)
		self.save_hyperparameters()

		self.feature_extractor = PLCLASP.load_from_checkpoint(clasp_checkpoint_path, map_location=clasp_map_location)
		self.feature_extractor.freeze()

		self.classifier = MLPLayers([self.feature_extractor._hparams.output_dim, num_classes], dropout=dropout)

	def forward(self, x:torch.Tensor) -> torch.Tensor:
		x = self.feature_extractor.encode_audio(x)
		x = F.normalize(x, dim=-1)
		x = self.feature_extractor.model.audio_transform(x)

		return self.classifier(x)

	def training_step(self, batch, batch_idx):
		_, loss, acc = self._shared_eval_step(batch, batch_idx)
		self.log('train_loss', loss, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		_, loss, acc = self._shared_eval_step(batch, batch_idx)
		metrics = {"val_acc": acc, "val_loss": loss}
		self.log_dict(metrics, prog_bar=True)

	def test_step(self, batch, batch_idx):
		_, loss, acc = self._shared_eval_step(batch, batch_idx)
		metrics = {"test_acc": acc, "test_loss": loss}
		self.log_dict(metrics)

	def _shared_eval_step(self, batch, batch_idx):
		labels, mels, _, _ = batch
		labels = labels[self.hparams.task]
		y_hat = self(mels)

		loss = F.cross_entropy(y_hat, labels)
		acc = accuracy(y_hat, labels)[0] / labels.size(0)

		return y_hat, loss, acc

	def configure_optimizers(self):
		return get_optimiser(self)

def get_optimiser(self):
	optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
	lr_scheduler = {
		'scheduler': CosineAnnealingWarmupRestarts(optimizer=optimizer, 
												T_max=self.hparams.LR_sheduler_T_max, 
												warmup_steps=self.hparams.LR_sheduler_warmup_steps, 
												max_lr=self.hparams.learning_rate, 
												min_lr=self.hparams.LR_sheduler_min_lr, 
												gamma=self.hparams.LR_sheduler_decay),
		'interval': self.hparams.lr_interval,
		'name': 'lr_scheduler',
		'monitor': 'valid_loss',
	}
	return [optimizer], [lr_scheduler]
