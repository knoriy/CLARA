import sys
sys.path.append('/fsx/knoriy/code/CLASP/clasp')

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
from utils import get_lists

import matplotlib.pyplot as plt

##############
# Non Critical imports
##############
from pprint import pprint

##############
# Zeroshot fn
##############

def calculate_average(data):
    num_items = len(data)
    if num_items == 0:
        return None

    keys = data[0].keys()

    avgs = {}
    for key in keys:
        sum_key = 0
        for d in data:
            sum_key += d[key]
        avgs[f"avg_{key}"] = sum_key / num_items

    return avgs


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

def run(model, zeroshot_weights, dataloader, metric_fn:MetricCollection):
    device = model.device
    model.eval()
    with torch.no_grad():
        metrics = []
        metric_fn = metric_fn.to(device)

        for batch in tqdm.tqdm(dataloader, desc='MiniBatch'):
            labels, mels, _, _ = batch
            labels = labels.to(device)
            mels = mels.to(device)

            ###############
            # Audio Features
            ###############
            audio_features = model.encode_audio(mels)
            audio_features = F.normalize(audio_features, dim=-1)
            audio_features = model.model.audio_transform(audio_features)

            ###############
            # Text Features
            ###############
            # text_features = model.encode_text(text)
            # text_features = F.normalize(text_features, dim=-1)
            # text_features = model.model.text_transform(text_features)

            # logits_per_audio = (audio_temp * (audio_features @ text_features.T))

            ###############
            # Get Temps
            ###############
            text_temp, audio_temp = model.get_temps()

            ###############
            # Zeroshot logits
            ###############
            logits_per_audio = (audio_temp * (audio_features @ zeroshot_weights.T))

            ###############
            # Get metrics
            ###############
            metric = metric_fn(logits_per_audio, labels)
            metrics.append(metric)

        avg_metric = calculate_average(metrics)

        predict = torch.argmax(logits_per_audio, dim=-1)
        # print("Labels", labels)
        plt.imshow(logits_per_audio.detach().cpu().numpy())
        # # print(logits_per_audio.detach().cpu().numpy())

    return avg_metric, labels, predict

def zeroshot_eval(model, classnames, templates, dataloader, language='en'):
    zeroshot_weights = zeroshot_classifier(model, classnames, templates, language=language)
    return run(model, zeroshot_weights, dataloader)

if __name__ == '__main__':
    from utils.tools import get_key
    ##############
    # Model
    ##############
    # model_path = "./logs/CLASP/kp0bwbfe/checkpoints/epoch=99-step=14700.ckpt"
    # model_path = "./logs/CLASP/bem5ka5l/checkpoints/epoch=243-step=417761.ckpt"
    model_path = "./logs/CLASP/3ayhc5jk/checkpoints/epoch=158-step=262555.ckpt"

    model = PLCLASP.load_from_checkpoint(model_path).to('cuda')

    ##############
    # DataModule
    ##############
    # dataset = ESC50TDM(
    #             test_urls=['/fsx/knoriy/raw_datasets/ESC-50-master/audio/'],
    #             batch_size = 1024,
    #             num_workers=12,
    #         )

    # templates = get_lists("./config/classification/sounds/esc-50/templates.txt")
    # with open("./config/classification/sounds/esc-50/minor_classes.json") as f:
    # 	classes = json.load(f)

    ##############
    # Gender
    ##############
    dataset = VoxCelebTDM(
                test_urls=['/fsx/knoriy/raw_datasets/VoxCeleb_gender/'],
                batch_size =8,
                num_workers=12,
            )
    templates = get_lists("./config/classification/gender/templates.txt")
    with open("./config/classification/gender/classes.json") as f:
        classes = json.load(f)

    ##############
    # Emotion
    ##############
    # dataset = EMNSTDM(
    # 			classes = './config/classification/emotion/emns/classes.json', 
    # 			test_urls=['/fsx/knoriy/raw_datasets/EMNS/metadata.csv'],
    # 			batch_size = 8,
    # 			num_workers=12,
    # 		)

    # templates = get_lists("./config/classification/emotion/emns/templates.txt")
    # with open("./config/classification/emotion/emns/classes.json") as f:
    # 	classes = json.load(f)

    ##############
    # Metric
    ##############
    num_classes = len(classes)
    metric = MetricCollection({})

    recall = []
    for top_k in [1, 2, 3, 5, 10]:
        if top_k > num_classes:
            break
        metric.add_metrics({
            f"rec@{top_k}":Recall(task='multiclass', num_classes=num_classes, top_k=top_k),
            })

    ##############
    # Run
    ##############
    dataset.setup()

    zeroshot_weights, all_texts = zeroshot_classifier(model, classes, templates)
    tops, labels, predicts = run(model, zeroshot_weights, dataset.test_dataloader(), metric)

    pprint(tops)
    # print(labels)
    # print(predicts)

    pprint([(get_key(classes, label.item()), get_key(classes, predict.item())) for label, predict in zip(labels, predicts)])