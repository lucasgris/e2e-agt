import ast
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

try:
    from models.modules import (
        conv_feature_extractor, 
        FCStringBlock, 
        LinearBlock, 
        GumbelVectorQuantizer
    )
except ImportError:
    from modules import (
        conv_feature_extractor, 
        FCStringBlock, 
        LinearBlock, 
        GumbelVectorQuantizer
    )

import logging
import warnings

logger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
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

        self.conv_feature_extractor = getattr(
            conv_feature_extractor, self.config["conv_feature_extractor"]["type"]
        )(self.config["conv_feature_extractor"]["config"])
        if self.config.get("conv_aggregation_input_dim", None) is None:
            if config["conv_feature_extractor"]["type"] == "ConvFeatureExtractor":
                inc = self.config["conv_feature_extractor"]["config"][
                    "conv_block_dims"
                ][-1]
            elif config["conv_feature_extractor"]["type"] == "ConvVHFFeatureExtractor":
                inc = (
                    self.config["conv_feature_extractor"]["config"][
                        "vertical_filter_conv"
                    ]["conv_block_dims"][-1]
                    + self.config["conv_feature_extractor"]["config"][
                        "horizontal_filter_conv"
                    ]["conv_block_dims"][-1]
                )
        else:
            inc = self.config["conv_aggregation_input_dim"]

        if self.config["conv_aggregation"] is not None:
            if self.config["conv_aggregation"] == "global_pooling":
                self.aggregator_notes = nn.AvgPool2d(
                    kernel_size=(1, inc), stride=(1, inc)
                )
            if inc != self.config["num_strings"]:
                if self.config["conv_aggregation"] is None:
                    raise ValueError(
                        f"Number of strings ({self.config['num_strings']}) must be equal to the last dimension of the conv_feature_extractor output ({inc}) or use conv_aggregation=='conv'."
                    )
            if self.config["conv_aggregation"] == "conv":
                if self.config["predict_notes"]:
                    self.aggregator_notes = nn.Conv2d(
                        in_channels=inc,
                        out_channels=1,
                        kernel_size=1,
                        stride=1,
                    )
                self.aggregator_strings = nn.Conv2d(
                    in_channels=inc,
                    out_channels=self.config["num_strings"],
                    kernel_size=1,
                    stride=1,
                )
            else:
                raise ValueError(
                    f"Unsupported conv_aggregation: {self.config['conv_aggregation']}"
                )

        if self.config.get("quantize_targets", False):
            raise ValueError("quantize_targets is not supported.")

        out_in = self.config["fc"]["in_channels"]

        if self.config["insert_onset_target_feature"]:
            logging.warning(
                "insert_onset_target_feature is deprecated and might not work properly."
            )
            self.fc_onset_target = nn.Linear(
                in_features=out_in + self.config["onset_target_feature_size"],
                out_features=out_in,
            )

        if self.config["insert_ffm"]:
            self.ffm_emb = LinearBlock(self.config["ffm_embedding_config"])
        else:
            self.ffm_emb = None

        if self.config["use_fc"]:
            fc = []
            for dim in self.config["fc_dims"]:
                fc.append(LinearBlock(self.config["fc"], out_in, dim))
                out_in = dim
            if self.config["fc_shared"]:
                self.fc_shared = nn.Sequential(*fc)
            else:
                self.fc = nn.Sequential(*fc)

        if self.config["use_self_attention"]:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=self.config["self_attention"]["embed_dim"],
                num_heads=self.config["self_attention"]["num_heads"],
                dropout=self.config["self_attention"]["dropout"],
                batch_first=True,
            )

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
        else:
            # Deprecated
            warnings.warn(
                "No output for notes or tablature. This is deprecated and might not work properly. Use FCStringBlock instead.",
                DeprecationWarning,
            )

            if self.config["output_type"] == "linear":
                self.output = nn.Linear(
                    in_features=out_in,
                    out_features=self.num_strings * self.config["output_per_string"],
                    bias=True,
                )
            elif self.config["output_type"] == "conv1d":
                self.output = nn.Conv1d(
                    in_channels=out_in,
                    out_channels=self.num_strings * self.config["output_per_string"],
                    kernel_size=1,
                )
            elif self.config["output_type"] == "conv2d":
                self.output = nn.Conv2d(
                    in_channels=out_in,
                    out_channels=self.num_strings * self.config["output_per_string"],
                    kernel_size=1,
                )

    def forward(
        self,
        x,
        ffm=None,
        onsets=None,
        freeze_feature_extractor=False,
        freeze_encoder=False,
        return_features=False,
        return_logits=False,
    ):
        features, residual, input_features = self.conv_feature_extractor(
            x, freeze=freeze_feature_extractor
        )

        output = {"features": input_features}

        # [B, F, C, T] -> [B, F, T, C]
        features = features.permute(0, 1, 3, 2)

        if (
            self.config["adaptative_pool_to_target_len"]
            and features.size(1) > self.config["target_len_frames"]
        ):
            features = F.adaptive_avg_pool2d(
                features, (self.config["target_len_frames"], features.size(3))
            )
        if (
            self.config["interpolate_target"]
            and features.size(1) < self.config["target_len_frames"]
        ):
            features = F.interpolate(
                features.unsqueeze(0),
                size=(
                    features.size(1),
                    self.config["target_len_frames"],
                    features.size(3),
                ),
            ).squeeze(0)

        attention_map = feat_notes = None
        if self.config["conv_aggregation"] is not None:
            if self.config["frame_wise"]:
                if self.config["conv_aggregation"] == "conv":
                    if self.config["predict_notes"]:
                        feat_notes = self.aggregator_notes(features)
                        feat_notes = feat_notes.squeeze(1)
                    feat_strings = self.aggregator_strings(features)
                    feat_strings = feat_strings.squeeze(1)
                elif self.config["conv_aggregation"] == "global_pooling":
                    if self.config["predict_notes"]:
                        feat_notes = self.aggregator_notes(features.transpose(1, 3))
                        feat_notes = feat_notes.transpose(1, 3).squeeze(3)
            else:
                # [B, F, C, T] -> [B, C] (segment classification)
                feat_notes = feat_notes.view(feat_notes.size(0), -1)
        else:
            feat_notes = feat_strings = features

        if self.config["insert_ffm"]:
            assert ffm is not None, str(
                "self.insert_ffm is set as true "
                "but no fret feature was provided in the "
                "forward pass."
            )
            ffm_emb = self.ffm_emb(ffm)
        else:
            ffm_emb = None

        if self.config["predict_notes"]:
            output_notes_emb = output_notes = self.output_notes(feat_notes)
            if self.config.get("predict_notes_blank", False):  # For MCTC
                output_notes_blank = self.output_notes_blank(feat_notes)
                output_notes_emb += output_notes_blank
                output_notes = torch.stack(
                    (
                        output_notes_blank,
                        output_notes,
                    ),
                    dim=1,
                )
                if self.config["output_notes"]["activation"] == "log_softmax":
                    output_notes = torch.log_softmax(output_notes, dim=1)
                elif self.config["output_notes"]["activation"] == "softmax":
                    output_notes = torch.softmax(output_notes, dim=1)
                elif self.config["output_notes"]["activation"] == "sigmoid":
                    output_notes = torch.sigmoid(output_notes)
        else:
            output_notes = None

        if freeze_encoder:
            logger.info("Freezing FC.")
            if self.config["use_fc"]:
                if self.config["fc_shared"]:
                    for param in self.fc_shared.parameters():
                        param.requires_grad = False
                else:
                    for param in self.fc.parameters():
                        param.requires_grad = False

        if self.config["use_fc"] and not self.config["fc_shared"]:
            z = self.fc(feat_strings)

            if self.config["use_self_attention"]:
                # B x S x T x C -> B x T x S x C -> B x T x (S x C)
                B, S, T, C = z.shape
                z = z.permute(0, 2, 1, 3).reshape(B, T, S * C)
                output["z"] = z.transpose(1, 2)
                z, attention_map = self.self_attention(z, z, z)
                output["c"] = z.transpose(1, 2)
                # B x T x (S x C) -> B x S x T x C
                z = z.reshape(B, T, S, C).permute(0, 2, 1, 3)
            else:
                attention_map = None
            output["attention_map"] = attention_map

        if self.config["use_fc_string_block"]:
            output_strings = []
            for s in range(self.config["num_strings"]):
                if self.config["use_fc"]:
                    if self.config["fc_shared"]:
                        output_string = self.fc_shared(z)
                    else:
                        output_string = z[:, s, :, :]
                else:
                    output_string = feat_strings[:, s, :, :]
                if self.config["insert_onset_target_feature"]:
                    # TODO: need test
                    assert onsets is not None, str(
                        "self.insert_onset_target_feature is set as true "
                        "but no onset target feature was provided in the "
                        "forward pass."
                    )
                    onsets = onsets.squeeze()
                    if len(onsets.size()) == 1:
                        onsets = onsets.unsqueeze(0)
                    output_string = torch.concat((output_string, onsets), dim=1)
                    output_string = self.fc_onset_target(output_string)
                if (
                    self.config["concat_notes_string_block"]
                    and self.config["predict_notes"]
                ):
                    if "detach_notes" in self.config and self.config["detach_notes"]:
                        output_string = torch.concat(
                            (output_string, output_notes_emb.detach()), dim=-1
                        )
                    else:
                        output_string = torch.concat(
                            (output_string, output_notes_emb), dim=-1
                        )
                if self.config["concat_residual_last_layers"]:
                    output_string = torch.concat((output_string, residual), dim=-1)
                if self.config["insert_ffm"]:
                    output_string = torch.concat((output_string, ffm_emb), dim=-1)
                output_strings.append(output_string)
            output_strings = torch.stack(output_strings, dim=1)
            if self.config["predict_onsets_and_frets"]:
                output_onsets = self.fc_string_block_onsets(
                    output_strings, return_logits=return_logits
                )
                output_frets = self.fc_string_block_frets(
                    output_strings, return_logits=return_logits
                )
            else:
                output_tab = self.fc_string_block_tab(
                    output_strings, return_logits=return_logits
                )
        else:
            if self.config["output_type"] == "linear":
                output_tab = self.output(features)
            elif self.config["output_type"] == "conv1d":
                output_tab = self.output(features.unsqueeze(2)).squeeze(2)
            elif self.config["output_type"] == "conv2d":
                output_tab = self.output(features.unsqueeze(2)).squeeze(2)
            output_tab = self.output(features).reshape(
                features.size(0),
                self.num_strings,
                features.size(1),
                self.config["output_per_string"],
            )
        output["ffm_emb"] = ffm_emb
        output["notes"] = output_notes
        if self.config["predict_onsets_and_frets"]:
            output["onsets"] = output_onsets
            output["frets"] = output_frets
        else:
            output["tab"] = output_tab
        return output


if __name__ == "__main__":
    from torchinfo import summary
    from omegaconf import OmegaConf

    config = OmegaConf.load("configs/cnn.yaml")
    model = CNN(config.model).to("cuda")
    print(model)
    summary(model, (2, *ast.literal_eval(config["input_size"])))
