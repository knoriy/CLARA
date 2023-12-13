import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tqdm
import torch
from torch.nn import functional as F

from torchmetrics import MetricCollection, Recall, Accuracy, Precision, AveragePrecision

from clara import PLCLARA
from utils import calculate_average
from eval.util import get_dataset, zeroshot_text

##############
# Non Critical imports
##############
from pprint import pprint

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
			audio_features = model.model.audio_transform(audio_features)
			audio_features = F.normalize(audio_features, dim=-1)

			###############
			# Text Features
			###############
			# text_features = model.encode_text(labels)
			# text_features = model.model.text_transform(text_features)
			# text_features = F.normalize(text_features, dim=-1)

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
	model = PLCLARA.load_from_checkpoint(args.model_path).to(device)

	##############
	# DataModule
	##############

	dataset, templates, classes = get_dataset(
		task = args.task, 
		dataset_name = args.dataset_name, 
		root_cfg_path = args.root_cfg_path, 
		batch_size = args.batch_size, 
		num_workers = args.num_workers
	)

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

	metric.add_metrics({f"AP":AveragePrecision(task='multiclass', num_classes=num_classes)})

	##############
	# Run
	##############
	dataset.setup()

	zeroshot_weights, all_texts = zeroshot_text(model, classes, templates)
	tops = run(model, zeroshot_weights, dataset.test_dataloader(), metric, args.task)

	return tops

if __name__ == '__main__':
	import argparse
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, help='Path to model')
	parser.add_argument('--task', type=str, choices=['texts', 'gender', 'emotion', 'age', 'sounds', 'speech'], help='Task to run')
	parser.add_argument('--dataset_name', type=str, required=True, help='if task is sounds or emotion, specify dataset name')
	parser.add_argument('--root_cfg_path', type=str, default='./config/', help='root path to config files')
	parser.add_argument('--top_k', type=int, default=[1,5,10], help='Top k metrics to use')
	parser.add_argument('--batch_size', type=int, default=8, help='Dataloader batch size')
	parser.add_argument('--num_workers', type=int, default=12, help='Dataloader number of workers')

	args = parser.parse_args()

	tops = main(args)

	pprint(tops)
