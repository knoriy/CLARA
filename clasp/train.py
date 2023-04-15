
import logging
pl_logger = logging.getLogger('pytorch_lightning')

import torch
import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.trainer import call


from typing import Dict, Set, Optional

from datamodule.td_tensored import TensoredTDM
from clasp import PLCLASP
from eval.zeroshot import zeroshot_eval
from utils import get_lists

class Trainer(pl.Trainer):
	def __init__(self, zeroshot_templates:str=None, zeroshot_classes:str=None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.zeroshot_templates = zeroshot_templates
		self.zeroshot_classes = zeroshot_classes

	def zeroshot(self, 
		model, 
		dataloaders:Optional[LightningDataModule] = None, 
		datamodule: Optional[LightningDataModule] = None,
		ckpt_path: Optional[str] = None,
	) -> tuple[float, float]:
		"""Zeroshot evaluation.""" 
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
			raise ValueError("trainer.zeroshot_classes is required, must be a path to a file containing a list of templates (one per line).")
		classes = get_lists(self.zeroshot_classes)

		pl_logger.info(f"Zeroshot evaluation with {len(classes)} classes and {len(templates)} templates")
		
		acc1, acc5 = zeroshot_eval(model, classes, templates, datamodule.test_dataloader())
		pl_logger.info(f"zeroshot accuracy@1: {acc1:.3f}, accuracy@5: {acc5:.3f}")

class MyLightningCLI(LightningCLI):

	def add_arguments_to_parser(self, parser):
		# parser.add_lightning_class_args(DataParallelStrategy, "ddp_Strategy")
		# parser.set_defaults({"ddp_Strategy.find_unused_parameters": "False"})
		pass

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
	import time
	pl_logger.info(f"Starting at {time.time()}")
	torch.set_float32_matmul_precision('medium')
	cli = MyLightningCLI(PLCLASP, TensoredTDM, trainer_class=Trainer, save_config_callback=None)
