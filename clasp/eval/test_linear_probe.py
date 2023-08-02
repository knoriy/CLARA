import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
import tqdm

from torchmetrics import MetricCollection, Recall, Accuracy

from clasp import LinearProbeCLASP
from utils import calculate_average
from eval.util import get_dataset


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
	parser.add_argument('--root_cfg_path', type=str, default='./config/', help='root path to config files')
	parser.add_argument('--task', type=str, choices=['gender', 'emotion', 'age', 'sounds'], help='Task to run')
	parser.add_argument('--dataset_name', type=str, required=False, help='if task is sounds or emotion, specify dataset name')
	parser.add_argument('--top_k', type=list[int], default=[1,2,3,5,10], help='Top k metrics to use')
	parser.add_argument('--batch_size', type=int, default=8, help='Dataloader batch size')
	parser.add_argument('--num_workers', type=int, default=12, help='Dataloader number of workers')
	parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='device to use for inference')

	args = parser.parse_args()

	tops, labels, classes = main(args)
	pprint.pprint(tops)
	