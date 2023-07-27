import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
import tqdm
from clasp import LinearProbeCLASP

import json

from datamodule import *
from utils import calculate_average, get_lists

from torchmetrics import MetricCollection, Recall, Accuracy
import tqdm

def run(model, dataloader, metric_fn:MetricCollection, limit_batches=-1, task='emotion'):
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
			# Forward
			###############
			logits = model(mels)
			if task == 'gender':
				logits = torch.argmax(logits, dim=1)

			###############
			# Get metrics
			###############
			metric = metric_fn(logits, labels)
			metrics.append(metric)

			if i+1 == limit_batches:
				break

		avg_metric = calculate_average(metrics)

	return avg_metric, labels


def main(args):
	##############
	# Model
	##############
	model = LinearProbeCLASP.load_from_checkpoint(args.model_path, clasp_map_location=args.device, clasp_checkpoint_path=args.clasp_path, map_location=args.device)

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
						batch_size = 1024,
						num_workers=12,
					)
			with open("./config/classification/sounds/esc-50/minor_classes.json") as f:
				classes = json.load(f)
		elif args.dataset_name == 'audioset':
			with open("./config/classification/sounds/audioset/classes.json") as f:
				classes = json.load(f)
			dataset = AudioSetTDM(
						test_urls=['s3://s-laion-audio/webdataset_tar/audioset/eval/'],
						classes=classes,
						batch_size = 1024,
						num_workers=12,
					)

	##########
	# Gender
	##########
	if args.task == 'gender':
		dataset = VoxCelebTDM(
					test_urls=['s3://s-laion/knoriy/VoxCeleb_gender/'],
					batch_size =8,
					num_workers=12,
				)
		with open("./config/classification/gender/classes.json") as f:
			classes = json.load(f)

	##########
	# Emotion
	##########
	if args.task == 'emotion':
		if args.dataset_name == 'emns':
			dataset = EMNSTDM(
    					root_data_path='s3://laion-west-audio/webdataset_tar/',
						batch_size = 8,
						num_workers=12,
					)

			with open("./config/classification/emotion/emns/classes.json") as f:
				classes = json.load(f)
		elif args.dataset_name == 'emov-db':
			dataset = EmovDBTDM(
    			root_data_path='s3://laion-west-audio/webdataset_tar/',
				batch_size =8,
				num_workers=12,
			)

			with open("./config/classification/emotion/emov-db/classes.json") as f:
				classes = json.load(f)
		elif args.dataset_name == 'crema-d':
			Warning("CREMA-D is not supported yet")
			dataset = CremaDTDM(
    			root_data_path='s3://laion-west-audio/webdataset_tar/',
				batch_size =8,
				num_workers=12,
			)

			with open("./config/classification/emotion/crema-d/classes.json") as f:
				classes = json.load(f)
		elif args.dataset_name == 'ravdess':
			dataset = RavdessTDM(
						root_data_path='s3://laion-west-audio/webdataset_tar/',
						batch_size = 8,
						num_workers = 12,
					)

			with open("./config/classification/emotion/ravdess/classes.json") as f:
				classes = json.load(f)

	##########
	# age
	##########
	if args.task == 'age':
		dataset = CommonVoiceTDM(
					test_urls=['s3://s-laion-audio/webdataset_tar/common_voice/test/'],
					batch_size = 8,
					num_workers=12,
				)

		with open("./config/classification/age/common_voice/classes.json") as f:
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
			})

	##############
	# Run
	##############
	dataset.setup()

	tops, labels = run(model, dataset.test_dataloader(), metric, task=args.task)

	return tops, labels, classes


if __name__ == '__main__':
	import argparse
	import pprint
	import torch

	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, help='Path to model with linear probe head')
	parser.add_argument('--clasp_path', type=str, help='Path to pretrained CLASP model')
	parser.add_argument('--task', type=str, choices=['gender', 'emotion', 'age', 'sounds'], help='Task to run')
	parser.add_argument('--dataset_name', type=str, choices=['esc50', 'audioset', 'emns', 'emov-db', 'crema-d', 'ravdess'], required=False, help='if task is sounds or emotion, specify dataset name')
	parser.add_argument('--top_k', type=list[int], default=[1,2,3,5,10], help='Top k metrics to use')
	parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='device to use for inference')

	args = parser.parse_args()

	tops, labels, classes = main(args)
	pprint.pprint(tops)
	