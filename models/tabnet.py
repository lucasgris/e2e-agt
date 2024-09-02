import ast
import random
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.modules.conv_feature_extractor import (
        RawConvFeatureExtraction,
        ConvFeatureExtractor,
    )
    from models.modules.context_network import (
        CausalContextNetwork,
        LSTMContextNetwork,
        TransformerContextNetwork,
    )
    from models.modules import FCStringBlock, LinearBlock, GumbelVectorQuantizer
except ImportError:
    from modules.conv_feature_extractor import (
        RawConvFeatureExtraction,
        ConvFeatureExtractor,
    )
    from modules.context_network import (
        CausalContextNetwork,
        LSTMContextNetwork,
        TransformerContextNetwork,
    )
    from modules import FCStringBlock, LinearBlock, GumbelVectorQuantizer

import logging

logger = logging.getLogger(__name__)


class TabNet(nn.Module):
    def __init__(self, config):
        super(TabNet, self).__init__()
        self.config = config

        if self.config["predict_tab"] and self.config["predict_onsets_and_frets"]:
            raise ValueError(
                "predict_tab and predict_onsets_and_frets cannot be both True."
            )

        if self.config["output_notes"]["out_dim"] is None:
            logger.info(
                "output_notes_out_dim not set in config, using note_octaves + 1 (silence) as default."
            )
            len_output_notes = len(self.config["note_octaves"]) + 1

        # Tasks
        if not self.config["frame_wise"]:
            logger.warning(
                "Non frame-wise mode (segment classification) is deprecated and might not work properly."
            )

        if self.config["feature_extractor"]["type"] == "RawConvFeatureExtractor":
            self.feature_extractor = RawConvFeatureExtraction(
                self.config["feature_extractor"]["config"]
            )
        elif self.config["feature_extractor"]["type"] == "ConvFeatureExtractor":
            self.feature_extractor = ConvFeatureExtractor(
                self.config["feature_extractor"]["config"]
            )
        else:
            raise ValueError(
                f'feature_extractor type {self.config["feature_extractor"]["type"]} not supported.'
            )

        # if self.config["mask_z_training"]["use_mask_emb"]:
        #     self.mask_emb = nn.Parameter(
        #         torch.FloatTensor(self.config["mask_z_training"]["dim"]).uniform_()
        #     )
        
        if self.config["time_masking"]:
            if self.config["time_masking"]["use_mask_emb"]:
                self.mask_emb = nn.Parameter(
                    torch.FloatTensor(self.config["context_network"]["config"]["in_channels"]).uniform_()
                )

        if self.config["quantize_targets"] and not self.config["projection"] is not None:
            raise ValueError(
                "quantize_targets is True, but projection is not defined."
            )
        if self.config["quantize_targets"]:
            self.quantizer = GumbelVectorQuantizer(self.config["quantizer"])
        
        if self.config["projection"] is not None:
            self.projection = nn.Linear(
                self.config["projection"]["in_dim"], self.config["projection"]["out_dim"]
            )

        if self.config["context_network"] is not None:
            if self.config["context_network"]["type"] == "CausalContextNetwork":
                self.context_network = CausalContextNetwork(
                    self.config["context_network"]["config"]
                )
            elif self.config["context_network"]["type"] == "LSTMContextNetwork":
                self.context_network = LSTMContextNetwork(
                    self.config["context_network"]["config"]
                )
            elif self.config["context_network"]["type"] == "TransformerContextNetwork":
                self.context_network = TransformerContextNetwork(
                    self.config["context_network"]["config"]
                )
            else:
                raise ValueError(
                    f'context_network type {self.config["context_network"]["type"]} not supported.'
                )

        if self.config.get("conv_aggregation", None) is not None:
            if self.config["conv_aggregation"] == "conv":
                if self.config["predict_notes"]:
                    self.aggregator_notes = nn.Conv2d(
                        in_channels=self.config["conv_aggregation_input_dim"],
                        out_channels=1,
                        kernel_size=1,
                        stride=1,
                    )
                self.aggregator_strings = nn.Conv2d(
                    in_channels=self.config["conv_aggregation_input_dim"],
                    out_channels=self.config["num_strings"],
                    kernel_size=1,
                    stride=1,
                )
            else:
                raise ValueError(
                    f"Unsupported conv_aggregation: {self.config['conv_aggregation']}"
                )

        if self.config["insert_ffm"]:
            self.ffm_emb = LinearBlock(self.config["ffm_embedding_config"])
        else:
            self.ffm_emb = None

        if self.config["use_fc"]:
            fc = []
            for dim in self.config["fc_dims"]:
                fc.append(LinearBlock(self.config["fc"], dim=dim))
                out_in = dim
            if self.config["fc_shared"]:
                self.fc_shared = nn.Sequential(*fc)
            else:
                self.fc = nn.Sequential(*fc)

        if self.config["predict_notes"]:
            self.output_notes = nn.Linear(
                self.config["output_notes"]["in_dim"], len_output_notes
            )
            if self.config.get("predict_notes_blank", False):  # For MCTC
                self.output_notes_blank = nn.Linear(
                    self.config["output_notes"]["in_dim"], len_output_notes
                )

            if self.config["concat_notes_string_block"]:
                out_in += len_output_notes

        if self.config["use_fc_string_block"]:
            if self.config["predict_onsets_and_frets"]:
                self.fc_string_block_onsets = FCStringBlock(
                    config=self.config["fc_string_block_onsets"],
                    input_dim=self.config["fc_string_block_onsets"]["in_channels"],
                    num_strings=self.config["num_strings"],
                )
                self.fc_string_block_frets = FCStringBlock(
                    config=self.config["fc_string_block_frets"],
                    input_dim=self.config["fc_string_block_frets"]["in_channels"],
                    num_strings=self.config["num_strings"],
                )
            elif self.config["predict_tab"]:
                self.fc_string_block_tab = FCStringBlock(
                    config=self.config["fc_string_block_tab"],
                    input_dim=self.config["fc_string_block_tab"]["in_channels"],
                    num_strings=self.config["num_strings"],
                )                

    def time_masking(self, hidden_states: torch.Tensor, lengths: torch.Tensor=None) -> tuple[torch.Tensor, torch.BoolTensor]:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`
            lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            Masked hidden states (torch.Tensor with shape `(B, L, D)`),
            Time mask (torch.BoolTensor with `(B, L)`)
            )
        """
        batch_size, num_steps, hidden_size = hidden_states.size()
        
        # non mask: 0, mask: 1
        time_mask_indices = torch.zeros(
            batch_size, num_steps + self.config["time_masking"]["num_mask_time_steps"],
            device=hidden_states.device, dtype=torch.bool
        )

        for batch in range(batch_size):
            time_mask_idx_candidates = list(range(int(lengths[batch])))
            k = int(self.config["time_masking"]["mask_time_prob"] * lengths[batch])
            start_time_idx_array = torch.tensor(
                random.sample(time_mask_idx_candidates, k=k), device=hidden_states.device
            )

            for i in range(self.config["time_masking"]["num_mask_time_steps"]):
                time_mask_indices[batch, start_time_idx_array+i] = 1

        time_mask_indices = time_mask_indices[:, :-self.config["time_masking"]["num_mask_time_steps"]]
        num_masks = sum(time_mask_indices.flatten())

        # Maks hidden states
        if self.config["time_masking"]["use_mask_emb"]:
            mask_values = self.mask_emb.unsqueeze(0).expand(num_masks, -1)
        else:
            mask_values = torch.zeros(num_masks, hidden_size, device=hidden_states.device)
        hidden_states[time_mask_indices] = mask_values

        return hidden_states, time_mask_indices

    def _mask_z_training(self, z):
        n_time_masks = self.config["mask_z_training"]["n_time_masks"]
        mask_length = self.config["mask_z_training"]["mask_length"]
        # z: B x C x T
        mask = torch.ones_like(z)
        n_time_masks = (
            random.choice(n_time_masks) if len(n_time_masks) > 1 else n_time_masks[0]
        )
        mask_index = torch.randint(0, z.size(2), (n_time_masks,))
        for i in mask_index:
            l = (
                random.randint(mask_length[0], mask_length[1])
                if len(mask_length) > 1
                else mask_length[0]
            )
            j = min(i, z.size(2) - 1)
            k = min(j + l, z.size(2) - 1)
            if self.config["mask_z_training"]["use_mask_emb"]:
                mask[:, :, j:k] = self.mask_emb.repeat(k - j, 1).T
            else:
                mask[:, :, j:k] = 0
        masked_z = z * mask
        return masked_z

    def forward(
        self,
        x,
        ffm=None,
        mask=None,
        freeze_feature_extractor=False,
        freeze_encoder=False,
        return_features=False,
        return_logits=False,
    ):
        # B x 1 x T -> B x C x T
        z, residual, features = self.feature_extractor(
            x, freeze=freeze_feature_extractor
        )

        z = z.squeeze(2).transpose(1, 2)
        
        lengths = torch.tensor([z.size(1)] * z.size(0), device=z.device)
        
        if self.config["quantize_targets"]:  
            q, ppl = self.quantizer(z)
            assert torch.isnan(q).sum() == 0, f"{q} ({z.shape} -> {q.shape}): q contains NaN."

        assert self.context_network is not None, "context_network is not defined."

        if self.config["time_masking"] is not None and self.training:
            masked_z, mask_indices = self.time_masking(z.clone(), lengths)
            residual = masked_z
            c = self.context_network(z.transpose(1, 2), mask=mask, freeze=freeze_encoder).transpose(1, 2)
        else:
            if self.config["time_masking"]:
                masked_z, mask_indices= self.time_masking(z.clone(), lengths)  # just for logging
            else:
                masked_z = mask_indices = None
            residual = z
            c = self.context_network(z.transpose(1, 2), mask=mask, freeze=freeze_encoder).transpose(1, 2)

        if self.config["projection"] is not None:
            # B x T x C
            o = self.projection(c)
        else:
            o = c

        if self.config["quantize_targets"]:
            output = {
                "features": features,
                "z": z,
                "masked_z": masked_z,
                "mask_indices": mask_indices,
                "c": c,
                "o": o, 
                "q": q,
                "ppl": ppl,
            }
        else:
            if self.config["projection"] is not None:
                y = self.projection(z)
            else:
                y = z
            output = {
                "features": features,
                "z": z,
                "masked_z": masked_z,
                "mask_indices": mask_indices,
                "c": c,
                "o": o, 
                "y": y,
            }
        if return_features:
            return output
        
        c = c.transpose(1, 2)
        if self.config["use_projection_for_logits"]:
            c = o

        # B x C x T -> B x 1 x C x T
        c = c.unsqueeze(1)
        # B x 1 x C x T -> B x S x C x T
        if self.config.get("conv_aggregation", None) is not None:
            if self.config["conv_aggregation"] == "conv":
                if self.config["predict_notes"]:
                    c_notes = self.aggregator_notes(c)
                x = self.aggregator_strings(c)
        else:
            x = c.repeat(1, self.config["num_strings"], 1, 1)

        # B x S x C x T -> B x S x T x C
        x = x.permute(0, 1, 3, 2)

        # B x C x T -> B x T x C
        residual = residual.permute(0, 2, 1)

        output["notes"] = None

        if self.config["use_fc"] and not self.config["fc_shared"]:
            x = self.fc(x)
        if self.config["use_fc_string_block"]:
            output_strings = []
            for s in range(self.config["num_strings"]):
                if self.config["use_fc"]:
                    if self.config["fc_shared"]:
                        output_string = self.fc_shared(x)
                    else:
                        output_string = x[:, s, :, :]
                else:
                    output_string = x[:, s, :, :]
                if self.config["concat_residual_last_layers"]:
                    output_string = torch.concat((output_string, residual), dim=-1)
                if self.config["insert_ffm"]:
                    raise NotImplementedError
                output_strings.append(output_string)
            output_strings = torch.stack(output_strings, dim=1)
            if self.config["predict_onsets_and_frets"]:
                output_onsets = self.fc_string_block_onsets(
                    output_strings, return_logits=return_logits
                )
                output_frets = self.fc_string_block_frets(
                    output_strings, return_logits=return_logits
                )
            elif self.config["predict_tab"]:
                output_tab = self.fc_string_block_tab(
                    output_strings, return_logits=return_logits
                )

        if self.config["predict_onsets_and_frets"]:
            output["onsets"] = output_onsets
            output["frets"] = output_frets
        elif self.config["predict_tab"]:
            output["tab"] = output_tab
            
        return output
