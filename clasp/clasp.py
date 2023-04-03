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

        x = F.relu(self.audio_fc1(x))

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
		# acc = self.acc_fn(*model_out)
		acc = torch.tensor(0.0)

		return model_out, loss, acc

	def predict_step(self, batch, batch_idx, dataloader_idx=0):
		model_out, loss, acc = self._shared_eval_step(batch, batch_idx)
		return model_out, loss, acc

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
		lr_scheduler = {
			'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10),
			# 'scheduler': CosineAnnealingWithWarmup(optimizer=optimizer, T_max=200, warmup_steps=20),
			'name': 'lr_scheduler',
			'monitor': 'valid_loss',
		}
		return [optimizer], [lr_scheduler]