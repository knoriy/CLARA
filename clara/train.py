import json
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torchmetrics import MetricCollection, Recall

from typing import Dict, Set, Optional

pl_logger = logging.getLogger('pytorch_lightning')

from eval.test_zeroshot import run as zeroshot_run
from eval.test_zeroshot import zeroshot_classifier
from utils import get_lists

class Trainer(pl.Trainer):
	def __init__(self, zeroshot_templates:str=None, zeroshot_classes:str=None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.zeroshot_templates = zeroshot_templates
		self.zeroshot_classes = zeroshot_classes

	def zeroshot(self, 
		model,
		task:str,
		top_k:list[int] = [1, 3, 5, 10],
		dataloaders:Optional[LightningDataModule] = None, 
		datamodule: Optional[LightningDataModule] = None,
		ckpt_path: Optional[str] = None,
	) -> tuple[float, float]:
		"""
		Zeroshot evaluation.
		""" 
		pl_logger.debug(f"{self.__class__.__name__}: trainer zeroshot stage")

		# if a datamodule comes in as the second arg, then fix it for the user
		if isinstance(dataloaders, LightningDataModule):
			datamodule = dataloaders
			dataloaders = None
		if dataloaders is not None and datamodule:
			raise MisconfigurationException("You cannot pass both `trainer.zeroshot(dataloaders=..., datamodule=...)`")

		# links data to the trainer
		datamodule.setup()
		self._data_connector.attach_data(model, datamodule=datamodule)

		if not self.zeroshot_templates:
			raise ValueError("trainer.zeroshot_templates is required, must be a path to a file containing a list of templates (one per line).")
		templates = get_lists(self.zeroshot_templates)

		if not self.zeroshot_classes:
			raise ValueError("trainer.zeroshot_classes is required, must be a path to a json file containing class labels.")
		with open(self.zeroshot_classes) as f:
			classes = json.load(f)

		pl_logger.info(f"Zeroshot evaluation with {len(classes)} classes and {len(templates)} templates")

		metric = MetricCollection({})

		num_classes = len(classes)

		for tk in top_k:
			if tk > num_classes:
				break
			metric.add_metrics({
				f"rec@{tk}":Recall(task=task, num_classes=num_classes, top_k=tk),
				})

		zeroshot_weights = zeroshot_classifier(model, classes, templates)
		tops = zeroshot_run(model, zeroshot_weights, datamodule.test_dataloader(), metric_fn=None)
		pl_logger.info(f"zeroshot: {tops}")

class MyLightningCLI(LightningCLI):

	@staticmethod 
	def subcommands() -> Dict[str, Set[str]]: 
		"""Defines the list of available subcommands and the arguments to skip.""" 
		return { 
			"fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"}, 
			"validate": {"model", "dataloaders", "datamodule"}, 
			"test": {"model", "dataloaders", "datamodule"}, 
			"predict": {"model", "dataloaders", "datamodule"}, 
			"zeroshot": {"model", "dataloaders", "zeroshot_templates"}, 
		}

if __name__ == '__main__':
	import datetime
	pl_logger.info(f"Starting at {datetime.datetime.now()}")

	torch.set_float32_matmul_precision('medium')

	cli = MyLightningCLI(
		trainer_class=Trainer, 
		save_config_callback=None,
	)
