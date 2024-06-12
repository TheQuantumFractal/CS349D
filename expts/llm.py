#!/usr/bin/env python3
"""
file: model.py
--------------
Transformer model for LLM experiment

Model architecture ripped off of:
    https://github.com/stanford-cs336/spring2024-assignment4-data/blob/master/cs336-basics/cs336_basics/model.py

Llama3-8B:
    https://huggingface.co/meta-llama/Meta-Llama-3-8B
TinyStories:
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
"""
from __future__ import annotations

import sys
import os
import json
import argparse
import logging
from typing import Optional
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wrapper
from simulation import FaultSimulator, train

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(message)s", datefmt="%I:%M:%S %p", stream=sys.stdout
)

DATA_DIR = "expts/data/"
OUTPUT_DIR = "expts/model/"


"""" MODEL ARCHITECTURE """


class RMSNorm(nn.Module):
    """
    This module implements root mean square layer normalization, as
    described in Eq. 4 of https://arxiv.org/abs/1910.07467

    Args:
        hidden_size: int
            Dimensionality of the input to normalize.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.

    Returns:
        FloatTensor of same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: FloatTensor of shape `(batch_size, *)`.
                The input to apply root mean square layer normalization on.

        Returns:
            FloatTensor of same shape as input
        """
        # NOTE: in practice, many implementations will
        # manually upcast the input to fp32 here to prevent overflow when you
        # square the input.
        # https://github.com/pytorch/pytorch/issues/66707
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x


class TransformerLM(nn.Module):
    """A Transformer language model.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: Optional[float], default is None.
            If given, apply dropout to the sum of the token and position embeddings
            as well as the output of each sub-layer, before it is added to the
            sub-layer input and normalized (section 5.4).

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the
        predicted unnormalized next-word distribution for each token.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        # Store the model configuration for serialization / deserialization
        self.config = {
            k: v
            for k, v in locals().items()
            if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    attn_pdrop=attn_pdrop,
                    residual_pdrop=residual_pdrop,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie the weights, since the paper mentions that "we share the same weight
        # matrix between the two embedding layers and the pre-softmax linear transformation"
        self.lm_head.weight = self.token_embeddings.weight
        self.residual_pdrop = residual_pdrop
        # report number of parameters
        logging.info("number of non-embedding parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embeddings.weight.numel()
        return n_params

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: LongTensor of shape `(batch_size, sequence_length)`.
                Input IDs for language modeling.

        Returns: A FloatTensor of shape
            (batch size, sequence_length, vocab_size) with the predicted unnormalized next-word
            distribution for each token.
        """
        _, sequence_length = x.size()
        # (batch size, sequence_length, d_model)
        # NOTE: paper mentions "In the embedding layers, we multiply those
        # weights by sqrt(d_model)", but we aren't doing that here.
        embedded_tokens = self.token_embeddings(x)

        # Shape: (1, sequence_length)
        positions = torch.arange(0, sequence_length, dtype=torch.long, device=x.device).unsqueeze(0)
        # (1, sequence_length, d_model)
        embedded_positions = self.position_embeddings(positions)
        # (batch size, sequence_length, d_model)
        x = embedded_tokens + embedded_positions
        if self.residual_pdrop:
            # (batch size, sequence_length, d_model)
            x = F.dropout(x, self.residual_pdrop)
        for layer in self.layers:
            # (batch size, sequence_length, d_model)
            x = layer(x)
        # (batch size, sequence_length, d_model)
        x = self.ln_final(x)
        # (batch size, sequence_length, vocab_size)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        """
        Args:
            x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
            top_k: int
                If provided, only sample from the `top_k` vocab items (by probability).
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            # Take the last `context_length` tokens if the input is
            # beyond the model's context length
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            # Get the logits from the model
            logits = self.forward(x)
            # Take the logits for the next token
            next_token_logits = logits[:, -1]
            # apply temperature scaling
            temperature_scaled_next_token_logits = next_token_logits / temperature
            # If top-k is provided, take the tokens with the highest score
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                # Get the score of the kth item that we kept---items with lower scores should be masked.
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
            next_token_probabilities = F.softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            # End generation if we see the EOS token ID
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model


class TransformerBlock(nn.Module):
    """A single Transformer layer.

    This implements a single layer of the Transformer, as described in section 3.1
    of the paper.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: Optional[float], default is None.
            If given, apply dropout to the output of each sub-layer, before it is added
            to the sub-layer input and normalized (section 5.4).

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_pdrop if attn_pdrop else 0.0,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=True,
        )
        self.ln1 = RMSNorm(d_model)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model)
        self.residual_pdrop = residual_pdrop

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                The input to process with the Transformer block.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, d_model)`.
        """
        # NOTE: this is a pre-norm Transformer, and differs from the original
        # description in the paper.
        # Apply the multi-head self-attention sublayer
        x_ln = self.ln1(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        x_attn = self.attn(
            x_ln, x_ln, x_ln, need_weights=False, attn_mask=causal_mask, is_causal=True
        )[0]
        if self.residual_pdrop is not None:
            x_attn = F.dropout(x_attn, self.residual_pdrop)
        attn_sublayer_output = x + x_attn

        # Apply the feed-forward sublayer
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        if self.residual_pdrop is not None:
            x_ffn = F.dropout(x_ffn, self.residual_pdrop)
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = F.gelu(x)
        x = self.w2(x)
        return x


class LLMDataLoader:
    def __init__(self, train_data_path, valid_data_path, batch_size, context_length):
        self.train_data = np.memmap(train_data_path, mode="r", dtype=np.int32)
        self.valid_data = np.memmap(valid_data_path, mode="r", dtype=np.int32)
        self.batch_size = batch_size
        self.context_length = context_length
        # self.train_len = len(self.train_data) - context_length
        self.train_len = 100000
        self.val_len = len(self.valid_data) - context_length

    def get_batch(self, split, device):
        # in the <1 epoch regime, doesn't matter if nodes overlap data
        if split == "train":
            dataset = self.train_data
        elif split == "val":
            dataset = self.valid_data
        starting_idxs = torch.randint(len(dataset) - self.context_length - 1, (self.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy((dataset[i : i + self.context_length]).astype(np.int64))
                for i in starting_idxs
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy((dataset[i + 1 : i + 1 + self.context_length]).astype(np.int64))
                for i in starting_idxs
            ]
        )
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--p-fail",
        default=0.0,
        type=float,
        help="Probability of a fault occurring in simulator",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=128256,
        help="Path to file with mapping from token to BPE index",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Context length to use when training language model",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="The dimensionality of the model embeddings and sublayer outputs.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help=(
            "The number of Transformer layers to use. "
            "`d_model` must be evenly divisible by `num_heads`."
        ),
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help=(
            "Number of heads to use in multi-headed attention. "
            "`d_model` must be evenly divisible by `num_heads`."
        ),
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=2048,
        help=("Dimensionality of the feed-forward inner layer (section 3.3)."),
    )
    parser.add_argument(
        "--attn-pdrop",
        type=float,
        default=0.1,
        help=("If given, drop-out the attention probabilities with this rate."),
    )
    parser.add_argument(
        "--residual-pdrop",
        type=float,
        default=0.1,
        help=(
            "If given, apply dropout to output of each sub-layer, before it is "
            "added to the sub-layer input and normalized (section 5.4)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=("Batch size to use during training."),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help=("Number of epochs to train for."),
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=50000,
        help="Number of training steps to perform",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=2000,
        help="Measure validation loss every `eval-interval` training steps",
    )

    args = parser.parse_args()

    world_size = 2  # only have 2 GPUs

    dataloader = LLMDataLoader(
        train_data_path=os.path.join(DATA_DIR, "tiny-train.npy"),
        valid_data_path=os.path.join(DATA_DIR, "tiny-valid.npy"),
        batch_size=args.batch_size,
        context_length=args.context_length,
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
    )
    faultsim = FaultSimulator(args.p_fail, seed=0xDEADBEEF)

    logging.info("starting training")
    mp.spawn(
        train,
        args=(
            world_size,
            deepcopy(dataloader),
            deepcopy(model),
            F.cross_entropy,
            "AdamW",
            "Cosine",
            faultsim,
            args.num_epochs,
            100,
            args.eval_interval,
            OUTPUT_DIR,
            "flexitrain",
            f"llm-p{args.p_fail}",
        ),
        nprocs=world_size,
        join=True,
    )
    logging.info("finished running %s", sys.argv[0])
