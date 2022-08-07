from utils.tools import get_mask_from_lengths, partial
from model.hifigan.models import Generator, MultiScaleDiscriminator
from .modules import VarianceAdaptor, Linear
from transformer import Encoder, Decoder
import torch.nn as nn
import torch
import json
import os



class JETSSynthesizer(nn.Module):
    def __init__(self, preprocess_config, model_configs, train_config):
        super(JETSSynthesizer, self).__init__()
        self.preprocess_config = preprocess_config
        self.synthesizer_config = model_configs[0]
        self.generator_config = model_configs[1]
        self.train_config = train_config
        
        self.speaker_emb = None
        self.generator_config["num_mels"] = self.synthesizer_config["transformer"]["decoder_hidden"]
        self.generator_config["gin_channels"] = self.synthesizer_config["transformer"]["encoder_hidden"]
        if self.synthesizer_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                self.synthesizer_config["transformer"]["encoder_hidden"],
            )

        self.encoder = Encoder(self.synthesizer_config)
        self.variance_adaptor = VarianceAdaptor(
            self.preprocess_config, self.synthesizer_config, self.train_config)
        self.decoder = Decoder(self.synthesizer_config)
        self.generator = Generator(self.generator_config)


    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None, 
        mel_lens=None,
        max_mel_len=None,
        cwt_spec_targets=None,
        cwt_mean_target=None,
        cwt_std_target=None,
        uv=None,
        e_targets=None, 
        attn_priors=None, 
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None, 
        gen=False, 
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            g = self.speaker_emb(speakers)
            output = output + g.unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks, 
            attn_h, 
            attn_s, 
            attn_logprob
        ) = self.variance_adaptor(
            output, 
            src_lens, 
            src_masks,
            mels, 
            mel_lens, 
            mel_masks, 
            max_mel_len, 
            cwt_spec_targets,
            cwt_mean_target,
            cwt_std_target,
            uv,
            e_targets, 
            attn_priors, 
            g, 
            p_control,
            e_control,
            d_control,
            step, 
            gen, 
        )
            
        output, mel_masks = self.decoder(output, mel_masks)
        
        if not gen:
            output, indices = partial(
                y=output.transpose(1,2), 
                segment_size=self.generator_config["segment_size"], 
                hop_size=self.preprocess_config["preprocessing"]["stft"]["hop_length"])
            output = output.transpose(1,2)
        else:
            indices = [None, None]
        
        wav = self.generator(output.transpose(1,2), g=g.unsqueeze(-1))

        return (
            wav,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            indices, 
            src_lens,
            mel_lens,
            attn_h, 
            attn_s, 
            attn_logprob
        )
    