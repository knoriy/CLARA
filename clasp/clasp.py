import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import numpy as np

from typing import Tuple, Union, Callable, Optional
from encoders.text_encoders import SimpleTransformer 
from encoders.audio_encoders import WhisperAudioEncoder, SimpleCNN, SimpleCNNLarge, Cnn1D10, Cnn1D12, resnet18, ResNeXt
from encoders.modules import PositionalEncoding, LayerNorm, MLPLayers
from loss import CLAPLoss, CLIPLoss

from scheduler import CosineAnnealingWithWarmup
from utils.accuracy import Accuracy, accuracy


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
                in_channels = self.hparm.text_encoder_width,
                out_channels = self.hparm.text_encoder_embedding,
                num_layers = self.hparm.text_encoder_layers,
                nhead = self.hparm.text_encoder_heads, 
                batch_first=True,
                )

        if self.audio_encoder == None:
            # self.audio_encoder = SimpleCNN(80, 1024)
            # self.audio_encoder = SimpleCNNLarge(80, 1024)
            # self.audio_encoder = Cnn1D10(80, 1024)
            # self.audio_encoder = Cnn1D12(80, 1024)
            # self.audio_encoder = resnet18(1024)
            self.audio_encoder = ResNeXt(5,12,1024, 2, 4)
            # self.audio_encoder = WhisperAudioEncoder(80, 1024, 1, 1, batch_first=True)

        # ------------
        # Text Layers
        # ------------
        self.text_embedding = nn.Embedding(self.hparm.vocab_size, self.hparm.text_encoder_embedding)
        self.positional_embedding = PositionalEncoding(self.hparm.text_encoder_embedding)
        self.ln_final = LayerNorm(self.hparm.text_encoder_width)
        self.text_transform = MLPLayers(units=[1024,1024], dropout=0.1)

        # ------------
        # Audio Layers
        # ------------
        self.audio_fc1 = nn.Linear(1024, 1024)
        self.audio_transform = MLPLayers(units=[1024,1024], dropout=0.1)

        # ------------
        # Other
        # ------------
        self.audio_tempeture = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text_tempeture = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(self, text:torch.Tensor):
        x = self.text_embedding(text)
        x = self.positional_embedding(x)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = torch.mean(x, 1)

        return x

    def encode_audio(self, audio:torch.Tensor):
        x = self.audio_encoder(audio)

        x1 = torch.mean(x, dim=2)
        x2, _ = torch.max(x, dim=2)
        x = x1 + x2

        x = F.leaky_relu(self.audio_fc1(x))

        return x

    def forward(self, text:torch.Tensor, audio:torch.Tensor):
        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        audio_features = self.encode_audio(audio)
        audio_features = F.normalize(audio_features, dim=-1)

        # Projection
        text_features = self.text_transform(text_features)
        audio_features = self.audio_transform(audio_features)

        return text_features, audio_features, self.text_tempeture.exp(), self.audio_tempeture.exp()

class PLCLASP(pl.LightningModule):
	def __init__(	self, 
					hidden_dim:int=128, 
					learning_rate:float=1e-3, 
					learning_rate_patience:int=10, 
					text_encoder_width:int=1024,
					text_encoder_embedding:int=1024,
					text_encoder_layers:int=1,
					text_encoder_heads:int=4,
					vocab_size:int=50373,
					n_mels:int=80,
					audio_encoder_embedding:int=1024,
					LR_sheduler_T_max:int=20,
					LR_sheduler_warmup_steps:int=20,
					LR_sheduler_min_lr:float=0.0,
					LR_sheduler_decay:float=1.0,
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
		texts, mels, _, _  = batch # torch.size([*, 123]), torch.size([*,80,1234])

		model_out = self(texts, mels)
		loss = self.loss_fn(*model_out)
		
		self.log('text_temp', model_out[2])
		self.log('audio_temp', model_out[3])
		self.log('train_loss', loss, prog_bar=True, sync_dist=True)

		return loss

	def validation_step(self, batch, batch_idx):
		_, loss, acc = self._shared_eval_step(batch, batch_idx)

		metrics = {"val_acc": acc, "val_loss": loss}
		self.log_dict(metrics, prog_bar=True, sync_dist=True)

	def test_step(self, batch, batch_idx):
		_, loss, acc = self._shared_eval_step(batch, batch_idx)

		metrics = {"test_acc": acc, "test_loss": loss}
		self.log_dict(metrics, sync_dist=True)

	def _shared_eval_step(self, batch, batch_idx):
		texts, mels, _, _  = batch # torch.size([*, 123]), torch.size([*,80,1234])
		model_out = self(texts, mels)

		loss = self.loss_fn(*model_out)
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
		checkpoint_path:str, 
		learning_rate:float=1e-3, 
		LR_sheduler_T_max:int=20,
		LR_sheduler_warmup_steps:int=20,
		LR_sheduler_min_lr:float=0.0,
		LR_sheduler_decay:float=1.0,
		*args, **kwargs
	) -> None:
		
		super().__init__(*args, **kwargs)
		self.save_hyperparameters()

		self.feature_extractor = PLCLASP.load_from_checkpoint(checkpoint_path)
		self.feature_extractor.freeze()

		self.classifier = MLPLayers([1024, 512, 128, num_classes])

	def forward(self, x:torch.Tensor) -> torch.Tensor:
		x = self.feature_extractor.encode_audio(x)
		x = F.normalize(x, dim=-1)
		x = self.feature_extractor.model.audio_transform(x)

		return self.classifier(x)

	def training_step(self, batch, batch_idx):
		labels, mels, _, _  = batch
		y_hat = self(mels)
		loss = F.cross_entropy(y_hat, labels)
		self.log('train_loss', loss, prog_bar=True, sync_dist=True)
		return loss

	def validation_step(self, batch, batch_idx):
		_, loss, acc = self._shared_eval_step(batch, batch_idx)
		metrics = {"val_acc": acc, "val_loss": loss}
		self.log_dict(metrics, prog_bar=True, sync_dist=True)

	def test_step(self, batch, batch_idx):
		_, loss, acc = self._shared_eval_step(batch, batch_idx)
		metrics = {"test_acc": acc, "test_loss": loss}
		self.log_dict(metrics, sync_dist=True)

	def _shared_eval_step(self, batch, batch_idx):
		labels, mels, _, _  = batch
		y_hat = self(mels)

		loss = F.cross_entropy(y_hat, labels)
		acc = accuracy(y_hat, labels)[0] / labels.size(0)

		return y_hat, loss, acc

	def configure_optimizers(self):
		return get_optimiser(self)

def get_optimiser(self):
	optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
	lr_scheduler = {
		'scheduler': CosineAnnealingWithWarmup(optimizer=optimizer, 
												T_max=self.hparams.LR_sheduler_T_max, 
												warmup_steps=self.hparams.LR_sheduler_warmup_steps, 
												max_lr=self.hparams.learning_rate, 
												min_lr=self.hparams.LR_sheduler_min_lr, 
												gamma=self.hparams.LR_sheduler_decay),
		'name': 'lr_scheduler',
		'monitor': 'valid_loss',
	}
	return [optimizer], [lr_scheduler]
