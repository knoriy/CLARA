import torch
import soundfile as sf
import tqdm
import librosa
import numpy as np
from clasp import LinearProbeCLASP

from utils import calculate_average

def run(model, zeroshot_weights, dataloader, metric_fn:MetricCollection, limit_batches=-1):
	device = model.device
	model.eval()
	with torch.no_grad():
		metrics = []

		metric_fn = metric_fn.to(device)

		for i, batch in enumerate(tqdm.tqdm(dataloader, desc='MiniBatch')):
			labels, mels, _, _ = batch
			labels = labels['texts'].to(device)
			mels = mels.to(device)

			###############
			# Forward
			###############
			y_hat = model(mels)

			###############
			# Get metrics
			###############
			metric = metric_fn(y_hat, labels)
			metrics.append(metric)
			if i == limit_batches:
				break

		avg_metric = calculate_average(metrics)

	return avg_metric, labels, y_hat

if __name__ == '__main__':
	model = LinearProbeCLASP.load_from_checkpoint('logs/CLASP/EmoV_DB_linearProbe/checkpoints/epoch=94-step=5795.ckpt', 
										map_location=torch.device('cpu'), 
										clasp_checkpoint_path='logs/CLASP/EmoV_DB_77acc_99epoch/checkpoints/epoch=99-step=3000.ckpt',
										clasp_map_location=torch.device('cpu')
										)
	