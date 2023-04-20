import sys
sys.path.append('/fsx/knoriy/code/CLASP/clasp')

import tqdm
import torch
import json
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from text.tokeniser import Tokeniser

from clasp import PLCLASP
from datamodule import ESC50TDM
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

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), -1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

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
    return zeroshot_weights

def run(model, zeroshot_weights, dataloader, topk=(1, 5)):
    device = model.device
    model.eval()
    with torch.no_grad():
        tops = len(topk) * [0] 
        n = 0

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
            # Get Accuracy
            ###############

            accs = accuracy(logits_per_audio, labels, topk=topk)
            for i, acc in enumerate(accs):
                tops[i] += acc
            n += mels.size(0)

        avg_accs = [acc/n for acc in tops]

        predict = torch.argmax(logits_per_audio, dim=-1)
        # print("Labels", labels)
        plt.imshow(logits_per_audio.detach().cpu().numpy())
        # # print(logits_per_audio.detach().cpu().numpy())

    return avg_accs, labels, predict

def zeroshot_eval(model, classnames, templates, dataloader, language='en'):
    zeroshot_weights = zeroshot_classifier(model, classnames, templates, language=language)
    return run(model, zeroshot_weights, dataloader)

if __name__ == '__main__':
    from utils.tools import get_key
    ##############
    # Model
    ##############
    model = PLCLASP()
    model = model.load_from_checkpoint("./logs/CLASP/kp0bwbfe/checkpoints/epoch=99-step=14700.ckpt").to('cuda')
    model = model.to("cuda")

    ##############
    # DataModule
    ##############
    tokenizer = Tokeniser()
    dataset = ESC50TDM(
                root_data_path=['/fsx/knoriy/raw_datasets/ESC-50-master/audio/'],
                batch_size = 16,
                num_workers=12,
            )
    dataset.setup()

    templates = get_lists("./config/classification/gender/templates.txt")
    # classes = get_lists("./config/classification/gender/classes.txt")

    with open("./config/classification/gender/classes.json") as f:
        classes = json.load(f)

    zeroshot_weights, all_texts = zeroshot_classifier(model, classes, templates)
    tops, labels, predicts = run(model, zeroshot_weights, dataset.test_dataloader(), topk=(1,3))

    pprint([(get_key(classes, label.item()), get_key(classes, predict.item())) for label, predict in zip(labels, predicts)])
    pprint(tops)