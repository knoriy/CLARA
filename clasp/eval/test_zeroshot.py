import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tqdm
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

from torchmetrics import MetricCollection, Recall, Accuracy, Precision
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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

def zeroshot_classifier(model, classnames, templates, language='en'):
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

def run(model, zeroshot_weights, dataloader, metric_fn:MetricCollection, task:str, limit_batches=-1):
	device = model.device
	model.eval()
	with torch.no_grad():
		metrics = []

		metric_fn = metric_fn.to(device)

		for i, batch in enumerate(tqdm.tqdm(dataloader, desc='MiniBatch')):
			labels, mels, _, _ = batch
			labels = labels[task].to(device)
			mels = mels.to(device)

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
			# text_features = model.encode_text(labels)
			# text_features = F.normalize(text_features, dim=-1)
			# text_features = model.model.text_transform(text_features)

			# logits_per_audio = (audio_temp * (audio_features @ text_features.T))

			###############
			# Zeroshot logits
			###############
			logits_per_audio = (audio_temp * (audio_features @ zeroshot_weights.T))

			###############
			# Get metrics
			###############
			metric = metric_fn(logits_per_audio, labels)
			metrics.append(metric)
			if i == limit_batches:
				break

		avg_metric = calculate_average(metrics)

	return avg_metric

def main(args):
	##############
	# Model
	##############
	model = PLCLASP.load_from_checkpoint(args.model_path).to(device)

	##############
	# DataModule
	##############

	##########
	# Sounds
	##########

	if args.task == 'sounds':
		if args.dataset_name == 'esc50':
			dataset = ESC50TDM(
						test_urls=['s3://s-laion-audio/webdataset_tar/esc50/test/'],
						batch_size = args.batch_size,
						num_workers = args.num_workers,
					)
			templates = get_lists(os.path.join(args.root_cfg_path , "classification/sounds/esc-50/templates.txt"))
			with open(os.path.join(args.root_cfg_path , "classification/sounds/esc-50/minor_classes.json")) as f:
				classes = json.load(f)
		elif args.dataset_name == 'audioset':
			templates = get_lists(os.path.join(args.root_cfg_path , "classification/sounds/audioset/templates.txt"))
			with open(os.path.join(args.root_cfg_path , "classification/sounds/audioset/classes.json")) as f:
				classes = json.load(f)
			dataset = AudioSetTDM(
						root_data_path='s3://laion-west-audio/webdataset_tar/',
						classes=classes,
						batch_size = args.batch_size,
						num_workers = args.num_workers,
					)

	##########
	# Gender
	##########
	if args.task == 'gender':
		dataset = VoxCelebTDM(
					test_urls=['s3://s-laion/knoriy/VoxCeleb_gender/'],
					batch_size = args.batch_size,
					num_workers = args.num_workers,
				)
		templates = get_lists(os.path.join(args.root_cfg_path , "classification/gender/templates.txt"))
		with open(os.path.join(args.root_cfg_path , "classification/gender/classes.json")) as f:
			classes = json.load(f)

	##########
	# Emotion
	##########
	if args.task == 'emotion':
		templates_path = os.path.join(args.root_cfg_path , f"classification/{args.task}/{args.dataset_name}/templates.txt")
		classes_path = os.path.join(args.root_cfg_path , f"classification/{args.task}/{args.dataset_name}/classes.json")

		if args.dataset_name == 'emns':
			dataset = EMNSTDM(
						root_data_path='s3://laion-west-audio/webdataset_tar/',
						batch_size = args.batch_size,
						num_workers = args.num_workers,
					)

			templates = get_lists(templates_path)
			with open(classes_path) as f:
				classes = json.load(f)
		elif args.dataset_name == 'emov-db':
			dataset = EmovDBTDM(
				root_data_path='s3://laion-west-audio/webdataset_tar/',
				batch_size = args.batch_size,
				num_workers = args.num_workers,
			)

			templates = get_lists(templates_path)
			with open(classes_path) as f:
				classes = json.load(f)
		elif args.dataset_name == 'crema-d':
			Warning("CREMA-D is not supported yet")
			dataset = CremaDTDM(
				root_data_path='s3://laion-west-audio/webdataset_tar/',
				batch_size = args.batch_size,
				num_workers = args.num_workers,
			)

			templates = get_lists(templates_path)
			with open(classes_path) as f:
				classes = json.load(f)

	##########
	# age
	##########
	if args.task == 'age':
		dataset = CommonVoiceTDM(
					test_urls=['s3://s-laion-audio/webdataset_tar/common_voice/test/'],
					batch_size = args.batch_size,
					num_workers = args.num_workers,
				)

		templates = get_lists(os.path.join(args.root_cfg_path , "classification/age/common_voice/templates.txt"))
		with open(os.path.join(args.root_cfg_path , "classification/age/common_voice/classes.json")) as f:
			classes = json.load(f)

	##############
	# Metric
	##############
	num_classes = len(classes)
	metric = MetricCollection({})

	for top_k in args.top_k:
		if top_k > num_classes:
			break
		metric.add_metrics({
			f"rec@{top_k}":Recall(task='multiclass', num_classes=num_classes, top_k=top_k),
			f"acc@{top_k}":Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k),
			f"pre@{top_k}":Precision(task='multiclass', num_classes=num_classes, top_k=top_k),
			})

	##############
	# Run
	##############
	dataset.setup()

	zeroshot_weights, all_texts = zeroshot_classifier(model, classes, templates)
	tops = run(model, zeroshot_weights, dataset.test_dataloader(), metric, args.task)

	return tops

if __name__ == '__main__':
	import argparse
	from utils.tools import get_key
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	

	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, help='Path to model')
	parser.add_argument('--task', type=str, choices=['gender', 'emotion', 'age', 'sounds'], help='Task to run')
	parser.add_argument('--root_cfg_path', type=str, help='root path to config files')
	parser.add_argument('--dataset_name', type=str, choices=['esc50', 'audioset', 'emns', 'emov-db', 'crema-d'], required=False, help='if task is sounds or emotion, specify dataset name')
	parser.add_argument('--top_k', type=int, default=[1,2,3,5,10], help='Top k metrics to use')
	parser.add_argument('--batch_size', type=int, default=8, help='Dataloader batch size')
	parser.add_argument('--num_workers', type=int, default=12, help='Dataloader number of workers')

	args = parser.parse_args()

	tops = main(args)

	pprint(tops)