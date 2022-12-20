import torch
import torch.nn as nn

from .decoder import get_decoder
from .encoder import get_encoder


class TrOMR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = get_encoder(args)
        self.decoder = get_decoder(args)
        self.args = args

    def forward(self, inputs, rhythms_seq, pitchs_seq, 
                lifts_seq, note_seq, mask, **kwargs):
        
        encoded = self.encoder(inputs)
        loss = self.decoder(rhythms_seq, pitchs_seq, 
                            lifts_seq, note_seq, 
                            context=encoded, mask=mask, **kwargs)
        return loss

    @torch.no_grad()
    def generate(self, x: torch.Tensor, temperature: float = 0.25):
        start_token = (
                torch.LongTensor([self.args.bos_token]*len(x))[:, None]
            ).to(x.device)
        nonote_token = (
                torch.LongTensor([self.args.nonote_token]*len(x))[:, None]
            ).to(x.device)

        out_lift, out_pitch, out_rhythm = self.decoder.generate(
            start_token, nonote_token , self.args.max_seq_len,
            eos_token=self.args.eos_token, context=self.encoder(x), 
            temperature=temperature)
        
        return out_lift, out_pitch, out_rhythm
