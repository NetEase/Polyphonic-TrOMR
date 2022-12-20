from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers.x_transformers import AttentionLayers, TokenEmbedding, AbsolutePositionalEmbedding, Decoder

class ScoreTransformerWrapper(nn.Module):
    def __init__(
        self,
        num_note_tokens,
        num_rhythm_tokens,
        num_pitch_tokens,
        num_lift_tokens,
        max_seq_len,
        attn_layers,
        emb_dim,
        l2norm_embed = False
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.lift_emb = TokenEmbedding(emb_dim, num_lift_tokens, l2norm_embed = l2norm_embed)
        self.pitch_emb = TokenEmbedding(emb_dim, num_pitch_tokens, l2norm_embed = l2norm_embed)
        self.rhythm_emb = TokenEmbedding(emb_dim, num_rhythm_tokens, l2norm_embed = l2norm_embed)
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed = l2norm_embed)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.init_()

        self.to_logits_lift = nn.Linear(dim, num_lift_tokens)
        self.to_logits_pitch = nn.Linear(dim, num_pitch_tokens)
        self.to_logits_rhythm = nn.Linear(dim, num_rhythm_tokens)
        self.to_logits_note = nn.Linear(dim, num_note_tokens)

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.lift_emb.emb.weight, std = 1e-5)
            nn.init.normal_(self.pitch_emb.emb.weight, std = 1e-5)
            nn.init.normal_(self.rhythm_emb.emb.weight, std = 1e-5)
            nn.init.normal_(self.pos_emb.emb.weight, std = 1e-5)
            return

        nn.init.kaiming_normal_(self.lift_emb.emb.weight)
        nn.init.kaiming_normal_(self.pitch_emb.emb.weight)
        nn.init.kaiming_normal_(self.rhythm_emb.emb.weight)

    def forward(
        self,
        rhythms,
        pitchs,
        lifts,
        mask = None,
        return_hiddens = True,
        **kwargs
    ):
        x = self.rhythm_emb(rhythms) + self.pitch_emb(pitchs) + self.lift_emb(lifts) + self.pos_emb(rhythms)
        x = self.project_emb(x)
        x, hiddens = self.attn_layers(x, mask = mask, return_hiddens = return_hiddens, **kwargs)
        select_hiddens = hiddens[0][3]
        
        x = self.norm(x)

        out_lifts = self.to_logits_lift(x)
        out_pitchs = self.to_logits_pitch(x)
        out_rhythms = self.to_logits_rhythm(x)
        out_notes = self.to_logits_note(x)
        return out_rhythms, out_pitchs, out_lifts, out_notes, x

def top_k(logits, thres = 0.9):
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class ScoreDecoder(nn.Module):
    def __init__(self, transoformer, noteindexes, num_rhythmtoken, ignore_index = -100, pad_value = 0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = transoformer
        self.max_seq_len = transoformer.max_seq_len

        note_mask = torch.zeros(num_rhythmtoken)
        note_mask[noteindexes] = 1
        self.note_mask = nn.Parameter(note_mask)

    @torch.no_grad()
    def generate(self, start_tokens, nonote_tokens, seq_len, eos_token = None, temperature = 1., filter_thres = 0.9, min_p_pow=2.0, min_p_ratio=0.02, **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out_rhythm = start_tokens
        out_pitch = nonote_tokens
        out_lift = nonote_tokens
        mask = kwargs.pop('mask', None)

        if mask is None:
            mask = torch.full_like(out_rhythm, True, dtype=torch.bool, device=out_rhythm.device)

        for _ in range(seq_len):
            mask = mask[:, -self.max_seq_len:]
            x_lift = out_lift[:, -self.max_seq_len:]
            x_pitch = out_pitch[:, -self.max_seq_len:]
            x_rhymthm = out_rhythm[:, -self.max_seq_len:]
            
            rhythmsp, pitchsp, liftsp, notesp, _ = self.net(x_rhymthm, x_pitch, x_lift,  mask=mask, **kwargs)
            
            filtered_lift_logits = top_k(liftsp[:, -1, :], thres = filter_thres)
            filtered_pitch_logits = top_k(pitchsp[:, -1, :], thres = filter_thres)
            filtered_rhythm_logits = top_k(rhythmsp[:, -1, :], thres = filter_thres)

            lift_probs = F.softmax(filtered_lift_logits / temperature, dim=-1)
            pitch_probs = F.softmax(filtered_pitch_logits / temperature, dim=-1)
            rhythm_probs = F.softmax(filtered_rhythm_logits / temperature, dim=-1)
            
            lift_sample = torch.multinomial(lift_probs, 1)
            pitch_sample = torch.multinomial(pitch_probs, 1)
            rhythm_sample = torch.multinomial(rhythm_probs, 1)

            out_lift = torch.cat((out_lift, lift_sample), dim=-1)
            out_pitch = torch.cat((out_pitch, pitch_sample), dim=-1)
            out_rhythm = torch.cat((out_rhythm, rhythm_sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (torch.cumsum(out_rhythm == eos_token, 1)[:, -1] >= 1).all():
                break

        out_lift = out_lift[:, t:]
        out_pitch = out_pitch[:, t:]
        out_rhythm = out_rhythm[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out_rhythm, out_pitch, out_lift

    def forward(self, rhythms, pitchs, lifts,notes, **kwargs):
        liftsi = lifts[:, :-1]
        liftso = lifts[:, 1:]
        pitchsi = pitchs[:, :-1]
        pitchso = pitchs[:, 1:]
        rhythmsi = rhythms[:, :-1]
        rhythmso = rhythms[:, 1:]
        noteso = notes[:, 1:]

        mask = kwargs.get('mask', None)
        if mask is not None and mask.shape[1] == rhythms.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        rhythmsp, pitchsp, liftsp, notesp, x = self.net(rhythmsi, pitchsi, liftsi, **kwargs) 
        
        loss_consist = self.calConsistencyLoss(rhythmsp, pitchsp, liftsp,notesp)
        loss_rhythm = F.cross_entropy(rhythmsp.transpose(1, 2), rhythmso, ignore_index = self.ignore_index)
        loss_pitch = F.cross_entropy(pitchsp.transpose(1, 2), pitchso, ignore_index = self.ignore_index)
        loss_lift = F.cross_entropy(liftsp.transpose(1, 2), liftso, ignore_index = self.ignore_index)
        loss_note = F.cross_entropy(notesp.transpose(1, 2), noteso, ignore_index = self.ignore_index)
        
        return dict(
            loss_rhythm=loss_rhythm,
            loss_pitch=loss_pitch,
            loss_lift=loss_lift,
            loss_consist=loss_consist,
            loss_note = loss_note
        )

    def calConsistencyLoss(self, rhythmsp, pitchsp, liftsp,notesp, gamma=10):
        notesp_soft = torch.softmax(notesp, dim=2)
        note_flag = notesp_soft[:,:,1]
        rhythmsp_soft = torch.softmax(rhythmsp, dim=2)
        rhythmsp_note = torch.sum(rhythmsp_soft * self.note_mask, dim=2)

        pitchsp_soft = torch.softmax(pitchsp, dim=2)
        pitchsp_note = torch.sum(pitchsp_soft[:,:,1:], dim=2)

        liftsp_soft = torch.softmax(liftsp, dim=2)
        liftsp_note = torch.sum(liftsp_soft[:,:,1:], dim=2)
        
        loss = gamma * (F.l1_loss(rhythmsp_note, note_flag) + 
                        F.l1_loss(note_flag, liftsp_note) + 
                        F.l1_loss(note_flag, pitchsp_note)) / 3.
        return loss
        
def get_decoder(args):
    return ScoreDecoder(
        ScoreTransformerWrapper(
            num_note_tokens=args.num_note_tokens,
            num_rhythm_tokens=args.num_rhythm_tokens,
            num_pitch_tokens=args.num_pitch_tokens,
            num_lift_tokens=args.num_lift_tokens,
            max_seq_len=args.max_seq_len,
            emb_dim=args.decoder_dim,
            attn_layers=Decoder(
                dim=args.decoder_dim,
                depth=args.decoder_depth,
                heads=args.decoder_heads,
                **args.decoder_args
            )),
        pad_value=args.pad_token,
        num_rhythmtoken = args.num_rhythmtoken,
        noteindexes = args.noteindexes)