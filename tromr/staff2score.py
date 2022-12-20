import os

import cv2
import torch
import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from transformers import PreTrainedTokenizerFast

from model import TrOMR

class StaffToScore(object):
    def __init__(self, args):
        self.args = args
        self.size_h = args.max_height
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrOMR(args)
        self.model.load_state_dict(torch.load(args.filepaths.checkpoint), strict=True)
        self.model.to(self.device)
        
        self.lifttokenizer = PreTrainedTokenizerFast(tokenizer_file=args.filepaths.lifttokenizer)
        self.pitchtokenizer = PreTrainedTokenizerFast(tokenizer_file=args.filepaths.pitchtokenizer)
        self.rhythmtokenizer = PreTrainedTokenizerFast(tokenizer_file=args.filepaths.rhythmtokenizer)
        self.transform = alb.Compose([
            alb.ToGray(always_apply=True),
            alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
            ToTensorV2(),
        ])
        
    def readimg(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        print(1)
        print(img.shape)
        
        if img.shape[-1] == 4:
            img = 255 - img[:,:,3]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError("Unsupport image type!")
        
        h, w, c = img.shape
        new_h = self.size_h
        new_w = int(self.size_h / h * w)
        new_w = new_w // self.args.patch_size * self.args.patch_size
        img = cv2.resize(img, (new_w, new_h))
        img = self.transform(image=img)['image'][:1]
        print(2)
        print(img.shape)
        print(img.dtype)
        return img

    def preprocessing(rgb):
        h, w, c = rgb.shape
        new_h = self.size_h
        new_w = int(self.size_h / h * w)
        new_w = new_w // self.args.patch_size * self.args.patch_size
        img = cv2.resize(rgb, (new_w, new_h))
        img = self.transform(image=img)['image'][:1]
        return img
    
    def detokenize(self, tokens, tokenizer):
        toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ''
                toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
                if toks[b][i] in (['[BOS]', '[EOS]', '[PAD]']):
                    del toks[b][i]
        return toks
    
    def predict_img2token(self, rgbimgs):
        if not isinstance(rgbimgs, list):
            rgbimgs = [rgbimgs]
        imgs = [self.preprocessing(item) for item in rgbimgs]
        imgs = torch.cat(imgs).float().unsqueeze(1)
        output = self.model.generate(imgs.to(self.device),
                                    temperature=self.args.get('temperature', .2))
        rhythm, pitch, lift = output
        return rhythm, pitch, lift
    
    def predict_token(self, imgpath):
        imgs = []
        if os.path.isdir(imgpath):
            for item in os.listdir(imgpath):
                imgs.append(self.readimg(os.path.join(imgpath, item)))
        else:
            imgs.append(self.readimg(imgpath))
        imgs = torch.cat(imgs).float().unsqueeze(1)
        output = self.model.generate(imgs.to(self.device),
                                    temperature=self.args.get('temperature', .2))
        rhythm, pitch, lift = output
        return rhythm, pitch, lift

    def predict(self, imgpath):
        rhythm, pitch, lift = self.predict_token(imgpath)
        
        predlift = self.detokenize(lift, self.lifttokenizer)
        predpitch = self.detokenize(pitch, self.pitchtokenizer)
        predrhythm = self.detokenize(rhythm, self.rhythmtokenizer)
        return predrhythm, predpitch, predlift

if __name__ == "__main__":
    from configs import getconfig
    args = getconfig('./workspace/config.yaml')
    handler = StaffToScore(args)
    predrhythm, predpitch, predlift = handler.predict('../examples/test2/dark_1962926-44.jpg')
    print(predrhythm)
    print(predpitch)
    print(predlift)