import os
import ast
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import fairseq
except ImportError:
    warnings.warn("fairseq not found. MERTModel model might not be available.")


import logging
logger = logging.getLogger(__name__)

try:
    from models.modules import FCStringBlock, LinearBlock
except ImportError:
    from modules import FCStringBlock, LinearBlock

try:
    try:
        from models.mert_fairseq import *
    except ImportError:
        from mert_fairseq import *
except ImportError:
    warnings.warn("mert_fairseq not found. MERTAGT model might not be available.")

class MERTAGT(nn.Module):

    def __init__(self, config):
        super(MERTAGT, self).__init__()
        self.config = config
        
        if self.config["predict_tab"] and self.config["predict_onsets_and_frets"]:
            raise ValueError("predict_tab and predict_onsets_and_frets cannot be both True.")
        
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
    
        if not config.path or not os.path.exists(config.path):
            raise ValueError(f"Path {config.path} does not exist. Please specify a valid path "
                              "for the pretrained model.")
        
        logger.info(f"Loading MERT model from {os.path.abspath(config.path)}")
        if config.get("use_fairseq", False):
            mert_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([config.path], arg_overrides=config.arg_overrides)
            mert_model = mert_model[0]
        else:
            raise NotImplementedError
            self.mert.load_state_dict(torch.load(config.path))
        
        if config.freeze_pretrained_model:
            logger.info("Freezing MERT model")
            mert_model.eval()
            for param in mert_model.parameters():
                param.requires_grad = False
        mert_model.remove_pretraining_modules()
        self.mert = mert_model

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
        
    def forward(self, audio, ffm=None, only_conv_feature=False, freeze_encoder=False, **kwargs):
        output = {}
        
        if freeze_encoder:
            self.mert.eval()
            for param in self.mert.parameters():
                param.requires_grad = False
        else:
            self.mert.train()
            for param in self.mert.parameters():
                param.requires_grad = True
                
        audio = audio.squeeze()
        padding_mask = None # TODO
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if only_conv_feature:
            output["features"] = features = self.mert.feature_extractor(audio) # TODO: check if this is correct
            return output
        else:
            res = self.mert(**{
                "source": audio,
                "padding_mask": padding_mask,
                "mask": False #self.apply_mask and self.training,
            }, features_only=True)
            x = output["x"] = res["x"]
            padding_mask = res["padding_mask"]

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
            output_notes_emb = output_notes = self.output_notes(x)
            if self.config.get("predict_notes_blank", False):  # For MCTC
                output_notes_blank = self.output_notes_blank(x)
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
        output["notes"] = output_notes

        # B x T x C -> B x 1 x T x C
        x = x.unsqueeze(1)
        # B x 1 x C x T -> B x S x C x T  # TODO, implement aggregation
        x = x.repeat(1, self.config["num_strings"], 1, 1)

        # # B x S x C x T -> B x S x T x C
        # x = x.permute(0, 1, 3, 2)
        
        if self.config["use_fc"] and not self.config["fc_shared"]:
            output["z"] = z = self.fc(x)
        if self.config["use_fc_string_block"]:
            output_strings = []
            for s in range(self.config["num_strings"]):
                if self.config["use_fc"]:
                    if self.config["fc_shared"]:
                        output_string = self.fc_shared(z)
                    else:
                        output_string = x[:, s, :, :]
                else:
                    output_string = x[:, s, :, :]
                if self.config["concat_notes_string_block"] and self.config["predict_notes"]:
                    if "detach_notes" in self.config and self.config["detach_notes"]:
                        output_string = torch.concat(
                            (output_string, output_notes_emb.detach()), dim=-1
                        )
                    else:
                        output_string = torch.concat(
                            (output_string, output_notes_emb), dim=-1
                        )
                if self.config["concat_residual_last_layers"]:
                    raise NotImplementedError
                if self.config["insert_ffm"]:
                    output_string = torch.concat((output_string, ffm_emb), dim=-1)
                output_strings.append(output_string)
            output_strings = torch.stack(output_strings, dim=1)
            if self.config["predict_onsets_and_frets"]:
                output_onsets = self.fc_string_block_onsets(output_strings)
                output_frets = self.fc_string_block_frets(output_strings)
            else:
                output_tab = self.fc_string_block_tab(output_strings)
                
        if self.config["predict_onsets_and_frets"]:
            output["onsets"] = output_onsets
            output["frets"] = output_frets
        else:
            output["tab"] = output_tab
            
        return output