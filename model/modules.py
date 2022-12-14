import os
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils.tools import get_mask_from_lengths, b_mas, pad, init_weights, get_padding
from .layers import Linear, Conv
from utils import pitch_utils

LRELU_SLOPE = 0.1


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config, train_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.aligner = AlignmentEncoder(preprocess_config, model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        
        self.encoder_hidden = model_config["transformer"]["encoder_hidden"]
        self.cwt_hidden_size = model_config["variance_predictor"]["cwt_hidden_size"]
        self.cwt_std_scale = model_config["variance_predictor"]["cwt_std_scale"]
        self.cwt_stats_out_dims = model_config["variance_predictor"]["cwt_stats_out_dims"]
        self.binarization_start_steps = train_config["duration"]["binarization_start_steps"]
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        self.preprocess_config = preprocess_config
        self.preprocess_config["preprocessing"]["pitch"]["cwt_scales"] = pitch_utils.get_lf0_cwt(np.ones(10))[1]
        assert self.pitch_feature_level == "frame_level"
        assert self.energy_feature_level == "frame_level"

        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            energy_min, energy_max = stats["energy"][:2]

        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )
        
        
        self.cwt_predictor = CWTPredictor(model_config)
        self.cwt_stats_predictor = CWTStatPredictor(model_config)
        
        self.pitch_embedding = nn.Embedding(n_bins, self.encoder_hidden)
        self.energy_embedding = nn.Embedding(n_bins, self.encoder_hidden)
        
    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
        These will no longer recieve a gradient.
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
        return torch.from_numpy(attn_out).to(attn.device)

    def get_pitch_embedding(self, x, memory, mel2ph, control, f0, uv):
        cwt = self.cwt_predictor(x) 
        cwt_stats = self.cwt_stats_predictor(memory[:, 0, :])  # [B, 2]
        cwt_spec, cwt_mean, cwt_std = cwt[:, :, :10], cwt_stats[:, 0], cwt_stats[:, 1]
        if f0 is None and uv is None:
            cwt_std = cwt_std * self.cwt_std_scale
            f0 = pitch_utils.cwt2f0_norm(
                cwt_spec, cwt_mean, cwt_std, mel2ph, self.preprocess_config["preprocessing"]["pitch"])
            uv = cwt[:, :, -1] > 0
        f0_denorm = pitch_utils.denorm_f0(f0, uv, self.preprocess_config["preprocessing"]["pitch"])
        f0_denorm = f0_denorm * control
        pitch = pitch_utils.f0_to_coarse(f0_denorm) 
        embedding = self.pitch_embedding(pitch)
        prediction = [cwt, cwt_mean, cwt_std]
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_len, 
        src_mask,
        mel=None, 
        mel_len=None, 
        mel_mask=None, 
        max_mel_len=None, 
        cwt_spec_target=None,
        cwt_mean_target=None,
        cwt_std_target=None, 
        uv=None, 
        energy_target=None, 
        attn_prior=None, 
        g=None, 
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None, 
        gen=False, 
    ):
        memory = x.clone()
        log_duration_prediction = self.duration_predictor(x, src_mask)
        
        if not gen:
            attn_s, attn_logprob = self.aligner(
                queries=mel, keys=x, mask=src_mask, attn_prior=attn_prior, g=g)
            attn_h = self.binarize_attention_parallel(attn_s, src_len, mel_len).detach()
            duration_rounded = attn_h.sum(2)[:, 0, :]
            if step < self.binarization_start_steps:
                x = torch.bmm(attn_s.squeeze(1),x)
            else:
                x, mel_len = self.length_regulator(x, duration_rounded, max_mel_len)
        else:
            attn_h, attn_s, attn_logprob = None, None, None
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_mel_len)
            mel_mask = get_mask_from_lengths(mel_len)
            
        mel2ph = pitch_utils.dur_to_mel2ph(duration_rounded, src_mask)[:, : mel_len.max()]
        
        if not gen:
            f0 = pitch_utils.cwt2f0_norm(
                cwt_spec_target, 
                cwt_mean_target, 
                cwt_std_target, 
                mel2ph, 
                self.preprocess_config["preprocessing"]["pitch"])
        else:
            f0 = None

        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, memory, mel2ph, p_control, f0, uv)
        x = x + pitch_embedding
        
        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, mel_mask, e_control)
        x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask, 
            attn_h, 
            attn_s, 
            attn_logprob
        )
    
    
class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel, 
                            padding=(self.kernel -1) // 2,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out
    
    
class CWTPredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(CWTPredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.hidden_size = model_config["variance_predictor"]["cwt_hidden_size"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]
        self.out_size = model_config["variance_predictor"]["cwt_out_dims"]
        
        self.preprojection = Linear(self.input_size, self.hidden_size)
        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.hidden_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel, 
                            padding=(self.kernel -1) // 2,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = Linear(self.filter_size, self.out_size)

    def forward(self, encoder_output):
        out = self.preprojection(encoder_output)
        out = self.conv_layer(out)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        return out
    
    
class CWTStatPredictor(nn.Module):
    def __init__(self, model_config):
        super(CWTStatPredictor, self).__init__()
        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.hidden_size = model_config["variance_predictor"]["cwt_hidden_size"]
        self.cwt_stats_out_dims = model_config["variance_predictor"]["cwt_stats_out_dims"]
        self.layer = nn.Sequential(
            Linear(self.input_size, self.hidden_size), 
            nn.ReLU(),
            Linear(self.hidden_size, self.hidden_size), 
            nn.ReLU(), 
            Linear(self.hidden_size, self.cwt_stats_out_dims)
        )
        
    def forward(self, x):
        return self.layer(x)
                        
    
    
class AlignmentEncoder(torch.nn.Module):
    """ Alignment Encoder for Unsupervised Duration Modeling """
    """From comprehensive transformer tts"""

    def __init__(self, preprocess_config, model_config):
        super(AlignmentEncoder, self).__init__()
        n_spec_channels = preprocess_config['preprocessing']['mel']['n_mel_channels']
        n_att_channels = model_config['variance_predictor']['filter_size']
        n_text_channels = model_config['transformer']['encoder_hidden']
        temperature = model_config['temperature']
        multi_speaker = model_config['multi_speaker']
        
        
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            Conv(
                n_text_channels,
                n_text_channels * 2,
                kernel_size=3, 
                padding=int((3 - 1) / 2), 
                bias=True,
                w_init='relu'
            ),
            nn.ReLU(),
            Conv(
                n_text_channels * 2,
                n_att_channels,
                kernel_size=1, 
                bias=True,
            ),
        )

        self.query_proj = nn.Sequential(
            Conv(
                n_spec_channels,
                n_spec_channels * 2,
                kernel_size=3, 
                padding=int((3 - 1) / 2), 
                bias=True,
                w_init='relu',
            ),
            nn.ReLU(),
            Conv(
                n_spec_channels * 2,
                n_spec_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(),
            Conv(
                n_spec_channels,
                n_att_channels,
                kernel_size=1, 
                bias=True,
            ),
        )

        if multi_speaker:
            self.key_spk_proj = Linear(n_text_channels, n_text_channels)
            self.query_spk_proj = Linear(n_text_channels, n_spec_channels)

    def forward(self, queries, keys, mask=None, attn_prior=None, g=None):
        """Forward pass of the aligner encoder.
        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): uint8 binary mask for variable length entries (should be in the T2 domain).
            attn_prior (torch.tensor): prior for attention matrix.
            speaker_embed (torch.tensor): B x C tnesor of speaker embedding for multi-speaker scheme.
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        if g is not None:
            keys = keys + self.key_spk_proj(g.unsqueeze(1).expand(
                -1, keys.shape[1], -1
            ))
            queries = queries + self.query_spk_proj(g.unsqueeze(1).expand(
                -1, queries.shape[-1], -1
            )).transpose(1, 2)
        keys_enc = self.key_proj(keys).transpose(1, 2)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries.transpose(1, 2)).transpose(1, 2)

        # Simplistic Gaussian Isotopic Attention
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            # print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape} mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            attn = self.log_softmax(attn) + torch.log(attn_prior.transpose(1,2)[:, None] + 1e-8)
            #print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.unsqueeze(2).permute(0, 2, 1).unsqueeze(2), -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob
    

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)
    

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
    
class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        if h.gin_channels != 0:
            self.cond = nn.Conv1d(h.gin_channels, h.upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
            
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()