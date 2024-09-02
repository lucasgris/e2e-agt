import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Optional
from itertools import groupby
from multiprocessing import Pool

import logging


class DiversityLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.G = config.num_code_vector_groups
        self.V = config.num_code_vectors_per_group
        self.a = config.loss_alpha

    def forward(self, perplexity):
        diversity_loss = self.diversity_loss(perplexity)
        return self.a * diversity_loss

    def diversity_loss(self, perplexity):
        """
        Args:
            perplexity (torch.Tensor): with shape `(G, V)`

        Returns:
            torch.Tensor with shape `(1)`
        """
        log_perplexity = torch.log(perplexity)
        entropy = torch.sum(perplexity*log_perplexity, dim=-1)
        diversity_loss = torch.sum(entropy) / (self.G * self.V)

        return diversity_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.contrastive_loss_temperature
        self.K = config.num_contrastive_loss_negative_samples
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, encoder_out, quantized_features, time_mask_indices):
        target_encoder_out = encoder_out[time_mask_indices]
        labels = quantized_features[time_mask_indices]

        # Number of targets per batch
        num_targets_per_batch = [int(time_mask_indices[i].sum()) for i in range(time_mask_indices.size(0))]

        # Make negative samples
        negative_samples = self.negative_sampler(labels, num_targets_per_batch)
        negative_samples = torch.cat([labels.unsqueeze(1), negative_samples], dim=1)

        contrastive_loss = self.contrastive_loss(target_encoder_out, labels, negative_samples)

        return contrastive_loss

    def contrastive_loss(
            self,
            targets,
            labels,
            negative_samples
    ):
        """
        Args:
            targets (torch.Tensor): with shape `(N, D)`
            labels (torch.Tensor): with shape `(N, D)`
            negative_samples (torch.Tensor): with shape `(N, K, D)`

        Returns:
            torch.Tensor with shape `(1)`
        """

        similarity = torch.exp(self.cos(targets, labels) / self.k)
        negative_similarity = torch.sum(torch.exp((self.cos(targets.unsqueeze(1), negative_samples) / self.k)), dim=1)

        contrastive_loss = -torch.log(similarity / negative_similarity).mean()

        return contrastive_loss

    def negative_sampler(self, label, num_targets_per_batch):
        """
        Args:
            label (torch.Tensor): with shape `(N, D)`
            num_targets_per_batch (list[int]): Number of targets per batch.

        Returns:
            torch.Tensor with shape `(N, K, D)'

        """
        negative_samples = []
        start_idx = 0
        for num_targets in num_targets_per_batch:
            negative_sample_candidate_indices = torch.arange(
                num_targets, device=label.device
            ).unsqueeze(0).repeat(num_targets, 1)

            diagonal = torch.eye(num_targets)

            # Pull yourself from the list of candidates. `(N, N)` -> `(N, N-1)`
            negative_sample_candidate_indices = negative_sample_candidate_indices[diagonal == 0].view(num_targets, -1)
            negative_sample_candidate_indices += start_idx

            where_negative_sample = (
                torch.tensor([i for i in range(num_targets) for _ in range(self.K)]),
                torch.tensor(
                    [random.sample(list(range(num_targets - 1)), k=self.K) for _ in range(num_targets)]).flatten()
            )

            # `(K * N)`
            negative_sample_indices = negative_sample_candidate_indices[where_negative_sample]

            negative_samples.append(label[negative_sample_indices])
            start_idx += num_targets

        negative_samples = torch.cat(negative_samples).view(label.size(0), self.K, -1)

        return negative_samples

class Wav2vec2Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(config)
        self.diversity_loss = DiversityLoss(config)
        self.a = config.loss_alpha

    def forward(self, encoder_out, quantized_features, perplexity, time_mask_indices):
        target_encoder_out = encoder_out[time_mask_indices]
        labels = quantized_features[time_mask_indices]

        # Number of targets per batch
        num_targets_per_batch = [int(time_mask_indices[i].sum()) for i in range(time_mask_indices.size(0))]

        # Make negative samples
        negative_samples = self.contrastive_loss.negative_sampler(labels, num_targets_per_batch)
        negative_samples = torch.cat([labels.unsqueeze(1), negative_samples], dim=1)
        contrastive_loss = self.contrastive_loss.contrastive_loss(target_encoder_out, labels, negative_samples)
        diversity_loss = self.diversity_loss.diversity_loss(perplexity)
        
        loss = contrastive_loss + self.a * diversity_loss

        return loss, contrastive_loss, diversity_loss
    
def ctc_alignment(logits, log_probs, targets, input_lengths, target_lengths, blank=0):
    if log_probs.requires_grad:
        # T, B, C -> B, C, T
        logits = logits.permute(1, 2, 0).to(targets.device)
        log_probs = F.log_softmax(logits, dim = 1)
        ctc_loss = F.ctc_loss(
            log_probs.permute(2, 0, 1),
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="sum"
        )
        (ctc_grad,) = torch.autograd.grad(
            ctc_loss, (logits,), retain_graph=True
        )
        temporal_mask = (
            torch.arange(
                logits.shape[-1], device=targets.device, dtype=input_lengths.dtype
            ).unsqueeze(0).to(targets.device)
            < input_lengths.unsqueeze(1).to(targets.device)
        )[:, None, :]
        alignment_targets = (log_probs.exp() * temporal_mask - ctc_grad).detach()
        ctc_loss_via_crossentropy = (-alignment_targets * log_probs).sum()
        return ctc_loss_via_crossentropy
    else:
        ctc_loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="mean"
        )
        return ctc_loss

class CTCLossByStringAlignment(nn.Module):
    def __init__(
        self, n=6, reduction="sum", ignore_empty=False, weights=None, **kwargs
    ):
        """
        n: number of CTC loss
        """
        super(CTCLossByStringAlignment, self).__init__()
        if reduction != "sum":
            raise ValueError("Only 'sum' reduction is supported")
        self.reduction = reduction
        self.ignore_empty = ignore_empty
        self.n = n
        if weights is not None:
            self.weights = weights
            assert len(weights) == n
        else:
            self.weights = None
        for kw in kwargs:
            logging.warning(
                f"CTCLossByStringAlignmentTargets: Ignoring keyword argument {kw}"
            )

    def forward(self, logits, log_probs, targets, input_lengths, target_lengths):
        losses = []
        for i in range(self.n):
            if targets[i].count_nonzero() == 0 and self.ignore_empty:
                losses.append(torch.zeros(1, dtype=torch.float32).to(targets[i].device))
                continue
            losses.append(
                ctc_alignment(
                    logits[i], log_probs[i], targets[i], input_lengths[i], target_lengths[i]
                ).to(targets[i].device)
            )
        if self.weights is not None:
            loss = loss * self.weights
        if self.reduction == "sum":
            loss = sum(losses)
        elif self.reduction == "mean":
            loss = sum(losses) / len(losses)
        else:
            loss = losses
        return torch.Tensor(loss), torch.Tensor(losses)

class CTCLossByString(nn.Module):
    def __init__(
        self, n=6, reduction="mean", ignore_empty=True, weights=None, **kwargs
    ):
        """
        n: number of CTC loss
        reduction: 'mean' | 'sum' | 'none'
        """
        super(CTCLossByString, self).__init__()
        self.reduction = reduction
        self.ignore_empty = ignore_empty
        if weights is not None:
            self.weights = weights
            assert len(weights) == n
        else:
            self.weights = None
        self.criterions = [
            torch.nn.CTCLoss(reduction=reduction, **kwargs) for i in range(n)
        ]

    def forward(self, logits, log_probs, targets, input_lengths, target_lengths):
        losses = []
        for i in range(len(self.criterions)):
            if targets[i].count_nonzero() == 0 and self.ignore_empty:
                losses.append(torch.zeros(1, dtype=torch.float32).to(targets[i].device))
                continue
            losses.append(
                self.criterions[i](
                    log_probs[i], targets[i], input_lengths[i], target_lengths[i]
                ).to(targets[i].device)
            )
        if self.weights is not None:
            loss = loss * self.weights
        if self.reduction == "sum":
            loss = torch.stack(losses).sum()
        elif self.reduction == "mean":
            loss = torch.stack(losses).mean()
        else:
            loss = losses
        return torch.Tensor(loss), torch.Tensor(losses)


class MultiResolutionCTCLossByString(nn.Module):
    def __init__(self, n=6, m=5, reduction="mean", ignore_empty=True, weights=None, **kwargs):
        """
        n: number of CTC loss
        reduction: 'mean' | 'sum' | 'none'
        """
        super(MultiResolutionCTCLossByString, self).__init__()
        self.n = n
        self.m = m
        self.ctc_losses = []
        for i in range(m):
            self.ctc_losses.append(
                CTCLossByString(n, reduction, ignore_empty, weights, **kwargs)
            )
        
    def forward(self, logits, log_probs, targets, input_lengths, target_lengths):
        losses = []
        string_losses = [[] for _ in range(self.n)]
        loss, string_loss = self.ctc_losses[0](logits, log_probs, targets, input_lengths, target_lengths)
        losses.append(loss)
        string_losses = [string_losses[j] + [string_loss[j]] for j in range(self.n)]
        cur_logits = logits
        for i in range(1, self.m):
            cur_logits = nn.functional.max_pool2d(cur_logits.permute(0, 2, 1, 3), (2, 1)).permute(0, 2, 1, 3)
            cur_log_probs = nn.functional.log_softmax(cur_logits, dim=2)
            cur_input_lengths = torch.stack(
                [
                    torch.full(
                        (cur_log_probs.shape[2],),
                        cur_log_probs.shape[1],
                        dtype=torch.long,
                    )
                    for _ in range(self.n)
                ]
            )
            loss, s_losses = self.ctc_losses[i](cur_logits, cur_log_probs, targets, cur_input_lengths, target_lengths)
            losses.append(loss)
            string_losses = [string_losses[j] + [s_losses[j]] for j in range(self.n)]
        return torch.stack(losses).sum(), [torch.stack(s_losses).sum() for s_losses in string_losses]

class MCTC(nn.Module):
    """Multi-label Connectionist Temporal Classification (MCTC) Loss
    Args:
    reduction='none'    No reduction / averaging applied to loss within this class.
                        Has to be done afterwards explicitly.

    For details see: https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#CTCLoss
    and
    C. Wigington, B.L. Price, S. Cohen: Multi-label Connectionist Temporal Classification. ICDAR 2019: 979-986
    """

    def __init__(self, reduction="mean", experimental_flatten_batch=False):
        super(MCTC, self).__init__()
        self.reduction = reduction
        self.experimental_flatten_batch = experimental_flatten_batch

    def _compute_one(self, log_probs, targets):
        char_unique, char_target = torch.unique(targets, dim=1, return_inverse=True)
        char_target = torch.remainder(char_target + 1, char_unique.size(1))
        char_unique = torch.roll(char_unique, 1, -1)
        char_targ_condensed = torch.tensor([t[0] for t in groupby(char_target)][1:])
        target_torch = char_targ_condensed.type(torch.cuda.LongTensor)
        target_length = torch.tensor(target_torch.size(0), dtype=torch.long)

        input_log_softmax = log_probs.unsqueeze(2)
        char_probs_nonblank = torch.matmul(
            1 - char_unique[:, 1:].transpose(0, -1),
            torch.squeeze(input_log_softmax[0, :, :, :]),
        ) + torch.matmul(
            char_unique[:, 1:].transpose(0, -1),
            torch.squeeze(input_log_softmax[1, :, :, :]),
        )
        char_probs_blank = input_log_softmax[1, :1, :, :].squeeze(2).squeeze(1)
        char_probs = torch.cat((char_probs_blank, char_probs_nonblank), dim=0)
        input_torch = char_probs.transpose(0, -1).type(torch.cuda.FloatTensor)
        input_length = torch.tensor(input_torch.size(0), dtype=torch.long)

        return torch.nn.functional.ctc_loss(
            input_torch,
            target_torch,
            input_length,
            target_length,
            reduction="none",
        )

    def forward(self, log_probs, targets):
        # TODO: this code is extremely slow and should be optimized using either torch.jit.script or batched operations
        """
        log_probs: (N, 2, C+1, T) tensor of log probabilities
            log_probs has two channels: 0 for blank, 1 for non-blank. The non-blank channel is used for the character (C) probabilities.
        targets: (N, C+1, L) tensor of targets, where L is the length of the target
        """
        if self.experimental_flatten_batch:
            raise NotImplementedError("Experimental flatten batch not implemented yet")
            batch_frames_len = log_probs.size(0) * log_probs.size(-1)
            loss = self._compute_one(
                log_probs.reshape(
                    batch_frames_len, log_probs.size(1), log_probs.size(2)
                ).permute(1, 2, 0),
                torch.cat([targets[i] for i in range(targets.size(0))], dim=1),
            )
        else:
            losses = []
            for i in range(log_probs.size(0)):
                l = self._compute_one(log_probs[i], targets[i])
                losses.append(l)
            if self.reduction == "mean":
                loss = sum(losses) / len(losses)
            elif self.reduction == "sum":
                loss = sum(losses)
            else:
                loss = losses
        return loss

class NoteFrameWiseLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        ignore_empty=False,
        frame_smoothing_kernel=None,
        compute_jit=False,
    ):
        super(NoteFrameWiseLoss, self).__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.ignore_empty = ignore_empty
        self.compute_jit = compute_jit
        if frame_smoothing_kernel is not None:
            self.frame_smoothing_kernel = (
                torch.tensor(list(frame_smoothing_kernel))
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
            )
            if self.frame_smoothing_kernel.shape[-1] % 2 == 0:
                raise ValueError(
                    "frame_smoothing_kernel should be an odd-sized kernel, got shape: {}".format(
                        self.frame_smoothing_kernel.shape
                    )
                )
            assert (
                len(self.frame_smoothing_kernel.shape) == 4
            ), "frame_smoothing_kernel should be a 4D kernel"
        else:
            self.frame_smoothing_kernel = None

    @torch.jit.script
    def _compute_note_loss_jit(
        output,
        target,
        ignore_empty: bool,
        frame_smoothing_kernel: Optional[torch.Tensor],
    ):
        """
        output: [batch_size, num_frames, num_classes]
        target: [batch_size, num_frames, num_classes]
        If not class_probs, it will compute the argmax of the target tensor to get the class indices
        """
        loss: List[torch.Tensor] = []
        if frame_smoothing_kernel is not None:
            target = F.conv2d(
                target.unsqueeze(1),
                frame_smoothing_kernel,
                padding=[
                    frame_smoothing_kernel.size(2) // 2,
                    frame_smoothing_kernel.size(3) // 2,
                ],
            ).squeeze(1)
        for t in range(output.shape[-1]):
            if ignore_empty and target[:, t].count_nonzero() == 0:
                continue
            target_t = target[:, t]
            loss_t = nn.functional.binary_cross_entropy_with_logits(
                output[:, t], target_t, reduction="mean"
            )
            loss.append(loss_t)
        return torch.stack(loss)

    def forward(self, output, target):
        """
        Args:
            output: [batch_size, num_frames, num_classes]
            target: [batch_size, num_frames, num_classes]
            If not class_probs, it will compute the argmax of the target tensor to get the class indices
        """
        if self.ignore_empty:
            if (
                len(target.shape) == 2
                and target.count_nonzero() == 0
                or len(target.shape) == 3
                and target.argmax(-1).count_nonzero() == 0
            ):
                raise RuntimeError(
                    "No non-empty target found. Try setting ignore_empty to False or check the target tensor"
                )
        if self.compute_jit:
            loss = self._compute_note_loss_jit(
                output,
                target,
                self.ignore_empty,
                self.frame_smoothing_kernel,
            )
        else:
            if self.frame_smoothing_kernel is not None:
                target = F.conv2d(
                    target.unsqueeze(1),
                    self.frame_smoothing_kernel,
                    padding=[
                        self.frame_smoothing_kernel.size(2) // 2,
                        self.frame_smoothing_kernel.size(3) // 2,
                    ],
                ).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                output.reshape(-1, output.size(-1)),
                target.reshape(-1, target.size(-1)),
                reduction="mean",
            )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

class CEStringFrameWiseLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        num_strings=6,
        weight_strings=None,
        label_smoothing=0,
        class_probs=False,
        frame_smoothing_kernel=None,
        compute_jit=False,
    ):
        super(CEStringFrameWiseLoss, self).__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.num_strings = num_strings
        self.label_smoothing = label_smoothing
        self.compute_jit = compute_jit
        if weight_strings is not None:
            self.weights = weight_strings
            assert len(weight_strings) == num_strings
        self.class_probs = class_probs
        if frame_smoothing_kernel is not None:
            self.frame_smoothing_kernel = (
                torch.tensor(list(frame_smoothing_kernel))
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
            )
            if self.frame_smoothing_kernel.shape[-1] % 2 == 0:
                raise ValueError(
                    "frame_smoothing_kernel should be an odd-sized kernel, got shape: {}".format(
                        self.frame_smoothing_kernel.shape
                    )
                )
            assert (
                len(self.frame_smoothing_kernel.shape) == 4
            ), "frame_smoothing_kernel should be a 4D kernel"
            if not class_probs:
                raise ValueError(
                    "frame_smoothing_kernel can only be used with class_probs=True"
                )
        else:
            self.frame_smoothing_kernel = None

    @torch.jit.script
    def _compute_fret_loss_jit(
        output,
        target,
        weights: Optional[List[float]],
        label_smoothing: float,
        class_probs: bool,
        frame_smoothing_kernel: Optional[torch.Tensor],
    ):
        """
        output: [batch_size, num_strings, num_frames, num_classes]
        target: [batch_size, num_strings, num_frames, num_classes]
        If not class_probs, it will compute the argmax of the target tensor to get the class indices
        """
        string_losses: List[torch.Tensor] = []
        if frame_smoothing_kernel is not None:
            target = (
                F.conv2d(
                    target.unsqueeze(1).reshape(
                        target.size(0) * target.size(1),
                        1,
                        target.size(2),
                        target.size(3),
                    ),
                    frame_smoothing_kernel.to(target.device),
                    padding=[
                        frame_smoothing_kernel.size(2) // 2,
                        frame_smoothing_kernel.size(3) // 2,
                    ],
                )
                .squeeze(1)
                .reshape(target.size(0), target.size(1), target.size(2), target.size(3))
            )
        for s in range(output.shape[1]):
            loss_s: List[torch.Tensor] = []
            target_s = target[:, s]
            for t in range(output.shape[2]):
                target_t = target_s[:, t]
                loss_s_t = nn.functional.cross_entropy(
                    output[:, s, t].float(),
                    target_t if class_probs else target_t.argmax(-1),
                    reduction="mean",
                    label_smoothing=label_smoothing,
                    weight=None,
                )
                loss_s.append(loss_s_t)
            if weights is not None:
                string_losses.append(torch.stack(loss_s).sum() * weights[s])
            else:
                string_losses.append(torch.stack(loss_s).sum())
        string_losses = torch.stack(string_losses)
        return string_losses.sum(), string_losses

    def forward(self, output, target):
        """
        Args:
            output: [batch_size, num_strings, num_frames, num_classes]
            target: [batch_size, num_strings, num_frames, num_classes]
            If not class_probs, it will compute the argmax of the target tensor to get the class indices
        """
        if self.compute_jit:
            fret_loss, string_losses = self._compute_fret_loss_jit(
                output,
                target,
                self.weights,
                self.label_smoothing,
                self.class_probs,
                self.frame_smoothing_kernel,
            )
        else:
            if self.frame_smoothing_kernel is not None:
                target = (
                    F.conv2d(
                        target.unsqueeze(1).reshape(
                            target.size(0) * target.size(1),
                            1,
                            target.size(2),
                            target.size(3),
                        ),
                        self.frame_smoothing_kernel,
                        padding=[
                            self.frame_smoothing_kernel.size(2) // 2,
                            self.frame_smoothing_kernel.size(3) // 2,
                        ],
                    )
                    .squeeze(1)
                    .reshape(
                        target.size(0), target.size(1), target.size(2), target.size(3)
                    )
                )
            string_losses = []
            for s in range(output.shape[1]):
                string_losses.append(
                    nn.functional.cross_entropy(
                        output[:, s].reshape(-1, output.size(-1)),
                        (
                            target[:, s].reshape(-1, target.size(-1))
                            if self.class_probs
                            else target[:, s].argmax(-1).reshape(-1)
                        ),
                        reduction="mean",
                        label_smoothing=self.label_smoothing,
                    )
                )
            string_losses = torch.stack(string_losses)
            fret_loss = string_losses.sum()

        if self.reduction == "mean":
            loss = fret_loss / self.num_strings
        elif self.reduction == "sum":
            loss = fret_loss
        else:
            loss = string_losses
        return loss, string_losses

class CEFrameWiseLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        label_smoothing=0,
        class_probs=False,
        frame_smoothing_kernel=None,
        compute_jit=False,
    ):
        super(CEFrameWiseLoss, self).__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.compute_jit = compute_jit
        self.class_probs = class_probs
        if frame_smoothing_kernel is not None:
            self.frame_smoothing_kernel = (
                torch.tensor(list(frame_smoothing_kernel))
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
            )
            if self.frame_smoothing_kernel.shape[-1] % 2 == 0:
                raise ValueError(
                    "frame_smoothing_kernel should be an odd-sized kernel, got shape: {}".format(
                        self.frame_smoothing_kernel.shape
                    )
                )
            assert (
                len(self.frame_smoothing_kernel.shape) == 4
            ), "frame_smoothing_kernel should be a 4D kernel"
        else:
            self.frame_smoothing_kernel = None

    @torch.jit.script
    def _compute_frame_loss_jit(
        output,
        target,
        label_smoothing: float,
        class_probs: bool,
        frame_smoothing_kernel: Optional[torch.Tensor],
    ):
        """
        output: [batch_size, num_frames, num_classes]
        target: [batch_size, num_frames, num_classes]
        If not class_probs, it will compute the argmax of the target tensor to get the class indices
        """
        losses: List[torch.Tensor] = []
        if frame_smoothing_kernel is not None:
            target = F.conv2d(
                target.unsqueeze(1),
                frame_smoothing_kernel,
                padding=[
                    frame_smoothing_kernel.size(2) // 2,
                    frame_smoothing_kernel.size(3) // 2,
                ],
            ).squeeze(1)
        for t in range(output.shape[1]):
            target_t = target[:, t]
            loss_t = nn.functional.cross_entropy(
                output[:, t].float(),
                target_t if class_probs else target_t.argmax(-1),
                reduction="mean",
                label_smoothing=label_smoothing,
                weight=None,
            )
            losses.append(loss_t)
        return torch.stack(losses)
    
    def forward(self, output, target):
        """
        Args:
            output: [batch_size, num_frames, num_classes]
            target: [batch_size, num_frames, num_classes]
            If not class_probs, it will compute the argmax of the target tensor to get the class indices
        """
        if self.compute_jit:
            frame_losses = self._compute_frame_loss_jit(
                output,
                target,
                self.label_smoothing,
                self.class_probs,
                self.frame_smoothing_kernel,
            )
        else:
            if self.frame_smoothing_kernel is not None:
                target = F.conv2d(
                    target.unsqueeze(1),
                    self.frame_smoothing_kernel,
                    padding=[
                        self.frame_smoothing_kernel.size(2) // 2,
                        self.frame_smoothing_kernel.size(3) // 2,
                    ],
                ).squeeze(1)
            frame_losses = []
            for t in range(output.shape[1]):
                frame_losses.append(
                    nn.functional.cross_entropy(
                        output[:, t],
                        target[:, t] if self.class_probs else target[:, t].argmax(-1),
                        reduction="mean",
                        label_smoothing=self.label_smoothing,
                    )
                )
            frame_losses = torch.stack(frame_losses)

        if self.reduction == "mean":
            loss = frame_losses.mean()
        elif self.reduction == "sum":
            loss = frame_losses.sum()
        else:
            loss = frame_losses
        return loss

class CEPretrainWithDiversityLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.ce_loss = CEFrameWiseLoss(config)
        self.diversity_loss = DiversityLoss(config)
        self.a = config.loss_alpha

    def forward(self, target_encoder_out, labels, perplexity, time_mask_indices):
        target_encoder_out = target_encoder_out[time_mask_indices]
        labels = labels[time_mask_indices]
        labels = torch.sigmoid(labels)
        # ce_loss = self.ce_loss(target_encoder_out, labels)
        frame_losses = []
        for t in range(target_encoder_out.shape[1]):
            frame_loss = nn.functional.cross_entropy(
                target_encoder_out[:, t], labels[:, t], 
                reduction="mean"
            )
            frame_losses.append(frame_loss)
        print([l.item() for l in frame_losses])
        ce_loss = torch.stack(frame_losses).sum()

        diversity_loss = self.diversity_loss.diversity_loss(perplexity)
        
        loss = ce_loss + self.a * diversity_loss

        return loss, ce_loss, diversity_loss

class CEPretrainLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ce_loss = CEFrameWiseLoss(**config)

    def forward(self, encoder_out, labels, time_mask_indices):
        # target_encoder_out = encoder_out[time_mask_indices]
        # labels = labels[time_mask_indices]
        # ce_loss = self.ce_loss(target_encoder_out, labels)

        target_encoder_out = encoder_out
        frame_losses = []
        for t in range(target_encoder_out.shape[1]):
            frame_loss = nn.functional.binary_cross_entropy_with_logits(
                target_encoder_out[:, t], labels[:, t], reduction="mean"
            )
            frame_losses.append(frame_loss)
        ce_loss = torch.stack(frame_losses).sum()

        return ce_loss

class FretOnsetByStringFrameWiseLoss(nn.Module):

    def __init__(
        self,
        reduction="sum",
        num_strings=6,
        weight_fret_onset=0.5,
        weight_strings=None,
        label_smoothing=0,
    ):
        super(FretOnsetByStringFrameWiseLoss, self).__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.num_strings = num_strings
        self.w = weight_fret_onset
        self.weight_strings = weight_strings
        self.label_smoothing = label_smoothing
        if weight_strings is not None:
            self.weights = weight_strings
            assert len(weight_strings) == num_strings
        else:
            self.weights = [1.0] * num_strings

    @torch.jit.script
    def _compute_fret_loss_jit(
        output, target, weights: Optional[List[float]]
    ):
        """
        output_fret: [batch_size, num_strings, num_frames, num_classes]
        target_fret: [batch_size, num_strings, num_frames] containing class indices or [batch_size, num_strings, num_frames, num_classes] for class probabilities
        """
        string_losses: List[torch.Tensor] = []
        for s in range(output.shape[1]):
            loss_s: List[torch.Tensor] = []
            for t in range(output.shape[2]):
                loss_s_t = nn.functional.cross_entropy(
                    output[:, s, t, :], target[:, s, t], reduction="mean", weight=None
                )
                loss_s.append(loss_s_t)
            if weights is not None:
                string_losses.append(torch.stack(loss_s).sum() * weights[s])
            else:
                string_losses.append(torch.stack(loss_s).sum())
        string_losses = torch.stack(string_losses)
        return string_losses.sum(), string_losses

    @torch.jit.script
    def _compute_onset_loss_jit(
        output, target, weights: Optional[List[float]]
    ):
        """
        output_onset: [batch_size, num_strings, num_frames]
        target_onset: [batch_size, num_strings, num_frames]
        """
        string_losses: List[torch.Tensor] = []
        for s in range(output.shape[1]):
            loss_s: List[torch.Tensor] = []
            for t in range(output.shape[2]):
                loss_s_t = nn.functional.cross_entropy(
                    output[:, s, t], target[:, s, t], reduction="mean"
                )
                loss_s.append(loss_s_t)
            if weights is not None:
                string_losses.append(torch.stack(loss_s).sum() * weights[s])
            else:
                string_losses.append(torch.stack(loss_s).sum())
        string_losses = torch.stack(string_losses)
        return string_losses.sum(), string_losses

    def forward(self, output_fret, target_fret, output_onset, target_onset, **kwargs):
        """
        Args:
            output_fret: [batch_size, num_strings, num_frames, num_classes]
            target_fret: [batch_size, num_strings, num_frames] containing class indices or [batch_size, num_strings, num_frames, num_classes] for class probabilities
            output_onset: [batch_size, num_strings, num_frames]
            target_onset: [batch_size, num_strings, num_frames]
            kwargs: additional arguments to pass to F.cross_entropy
        """
        fret_loss, fret_string_losses = self._compute_fret_loss_jit(
            output_fret, target_fret, self.weights
        )
        onset_loss, onset_string_losses = self._compute_onset_loss_jit(
            output_onset, target_onset, self.weights
        )
        loss = self.w * fret_loss + (1 - self.w) * onset_loss
        return loss, fret_loss, onset_loss, fret_string_losses, onset_string_losses


class ContrastiveByStringFrameWiseLoss(nn.Module):
    def __init__(self, temperature=1.0, reduction="mean", num_strings=6):
        super(ContrastiveByStringFrameWiseLoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.sim = nn.CosineSimilarity(dim=-1)
        self.num_strings = num_strings

    @torch.jit.script
    def _compute_contrastive_loss_jit(
        x,
        positive,
        distractors,
        temperature: float,
        reduction: str,
    ):
        """
        x: [B, S, T, D]
        positive: [B, S, T, D]
        distractors: [B, N, S, T, D]
        """
        string_losses: List[torch.Tensor] = []
        for s in range(x.shape[1]):
            loss_s: List[torch.Tensor] = []
            for t in range(x.shape[2]):
                anchor = x[:, s, t, :]
                positive_sample = positive[:, s, t, :]
                negative_samples = distractors[:, :, s, t, :]
                anchor_positive_similarity = (
                    nn.functional.cosine_similarity(anchor, positive_sample, dim=-1)
                    / temperature
                )
                anchor_negative_similarity = (
                    nn.functional.cosine_similarity(
                        anchor.unsqueeze(1).expand(-1, negative_samples.shape[1], -1),
                        negative_samples,
                        dim=-1,
                    )
                    / temperature
                )
                logits = torch.cat(
                    [
                        anchor_positive_similarity.unsqueeze(1),
                        anchor_negative_similarity,
                    ],
                    dim=1,
                )
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(
                    logits.device
                )
                loss_s_t = nn.functional.cross_entropy(logits, labels, reduction="mean")
                loss_s.append(loss_s_t)
            string_losses.append(torch.stack(loss_s).sum())
        string_losses = torch.stack(string_losses)
        if reduction == "mean":
            loss = string_losses.mean()
        elif reduction == "sum":
            loss = string_losses.sum()
        else:
            loss = string_losses
        return loss, string_losses

    def forward(self, x, positive, distractors):
        """
        Args:
            x: [B, S, T, D]
            positive: [B, S, T, D]
            distractors: [B, N, T, D]
        """
        loss, string_losses = self._compute_contrastive_loss_jit(
            x, positive, distractors, self.temperature, self.reduction
        )
        return loss, string_losses

