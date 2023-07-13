import tqdm
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

from torchmetrics import MetricCollection, Recall

from text.tokeniser import Tokeniser

from clasp import PLCLASP
from datamodule import *
from text.tokeniser import Tokeniser
from utils import get_lists, calculate_average

##############
# Non Critical imports
##############
from pprint import pprint

##############
# Zeroshot fn
##############

def zeroshot_text(model, classnames, templates, language='en'):
	tokenizer = Tokeniser()
	device = model.device
	with torch.no_grad():
		zeroshot_weights = []
		all_texts = []
		for classname in classnames:
			texts = [torch.tensor(tokenizer.encode(template.format(classname), language)) for template in templates]
			texts = pad_sequence(texts).T.contiguous().to(device)
			all_texts.append(texts)
			class_embeddings = model.encode_text(texts)
			class_embedding = F.normalize(class_embeddings, dim=-1)
			class_embedding = class_embedding.mean(dim=0)
			class_embedding = model.model.text_transform(class_embedding)
			class_embedding /= class_embedding.norm()
			zeroshot_weights.append(class_embedding)
		zeroshot_weights = torch.stack(zeroshot_weights).to(device)
	return zeroshot_weights, all_texts

##############
# Main
##############

def run(model, zeroshot_weights, dataloader, metric_fn:MetricCollection, limit_batches=-1, task='emotion'):
	device = model.device
	model.eval()
	with torch.no_grad():
		metrics = []

		metric_fn = metric_fn.to(device)

		for i, batch in enumerate(tqdm.tqdm(dataloader, desc='MiniBatch')):
			labels, mels, _, _ = batch
			labels = labels['texts'].to(device)
			mels = mels.to(device)

			batch_size = mels.size(0)

			###############
			# Get Temps
			###############
			text_temp, audio_temp = model.get_temps()

			###############
			# Audio Features
			###############
			audio_features = model.encode_audio(mels)
			audio_features = F.normalize(audio_features, dim=-1)
			audio_features = model.model.audio_transform(audio_features)

			###############
			# Text Features
			###############
			text_features = model.encode_text(labels)
			text_features = F.normalize(text_features, dim=-1)
			text_features = model.model.text_transform(text_features)


			###############
			# logits
			###############
			logits_per_audio = (audio_temp * (audio_features @ text_features.T))

			###############
			# Get metrics
			###############
			targets = torch.eye(logits_per_audio.size(0)).to(device)
			indexes = torch.range(i*batch_size, (i*batch_size-1)+batch_size, dtype=torch.int64).repeat(batch_size, 1).T

			metric_fn.update(logits_per_audio, targets, indexes)
		
			if i == limit_batches:
				break
		avg_metric = metric_fn.compute()

	return avg_metric

if __name__ == '__main__':
	import argparse
	import torch
	device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
	from clasp import PLCLASP

	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, help='Path to model with linear probe head')
	parser.add_argument('--task', type=str, choices=['texts', 'gender', 'emotion', 'age', 'sounds'], help='Task to run')
	parser.add_argument('--dataset_name', type=str, choices=['esc50', 'audioset', 'emns', 'emov-db'], required=False, help='if task is sounds or emotion, specify dataset name')
	parser.add_argument('--top-k', type=list[int], default=[1], help='Top k metrics to use')

	args = parser.parse_args()

	model = PLCLASP.load_from_checkpoint(args.model_path, map_location=device)

	##############
	# DataModule
	##############

	dataset = CremaDTDM(
		root_data_path='s3://laion-west-audio/webdataset_tar/',
		batch_size =5,
		num_workers=12,
	)
	dataset.setup()

	templates = get_lists("../config/classification/emotion/crema-d/templates.txt")
	with open("../config/classification/emotion/crema-d/classes.json") as f:
		classes = json.load(f)


	templates = ['{}']


	from torchmetrics.retrieval import RetrievalMAP

	metric_fn = RetrievalMAP(top_k=2)

	num_classes = len(classes)
	metric = MetricCollection({})

	for top_k in [1,2,3,4,5,6,7,8,9,10]:
		if top_k > num_classes:
			break
		metric.add_metrics({
			f"mAP@{top_k}":RetrievalMAP(top_k=top_k),
			})

	# zeroshot_weights, all_texts = zeroshot_classifier(model, classes, templates)

	run(model, None, dataset.test_dataloader(), metric, limit_batches=-1, task='emotion')


	