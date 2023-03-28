import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from text.tokeniser import Tokeniser

import tqdm

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), -1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def zeroshot_classifier(model, classnames, templates, language='en'):
    tokenizer = Tokeniser()
    device = model.device
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [torch.tensor(tokenizer.encode(template.format(classname), language)) for template in templates]
            texts = pad_sequence(texts).T.contiguous().to(device)
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights).to(device)
    return zeroshot_weights

def run(model, zeroshot_weights, dataloader):
    model.eval()
    with torch.no_grad():
        top1, top5, n = 0, 0, 0
        for batch in tqdm.tqdm(dataloader, desc='MiniBatch'):
            text, mels, _, _ = batch
            audio_features = model.encode_audio(mels)
            audio_features = F.normalize(audio_features, dim=-1)
            text_temp, audio_temp = model.get_temps()

            logits_per_audio = (audio_temp * (audio_features @ zeroshot_weights.T))

            # measure accuracy
            labels = torch.arange(text.shape[0], dtype=torch.long)

            acc1, acc5 = accuracy(logits_per_audio, labels, topk=(1, 4))
            top1 += acc1
            top5 += acc5
            n += mels.size(0)

        top1 = (top1 / n)
        top5 = (top5 / n)
    return top1, top5

def zeroshot_eval(model, classnames, templates, dataloader, language='en'):
    zeroshot_weights = zeroshot_classifier(model, classnames, templates, language=language)
    return run(model, zeroshot_weights, dataloader)