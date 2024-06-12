"""
file: llm_util.py
----------------
Utility functions for LLM experiment
"""

import os
import re
import logging
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

DATA_DIR = "expts/data/"


def tokenize_tiny_llama3():
    """
    Tokenize TinyStories dataset using llama3 tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained("llama3-8b")
    TINY_EOS = "<|endoftext|>"
    LLAMA3_EOS = "<|end_of_text|>"

    f = open(os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt"), "r")
    train_data = []
    for line in tqdm(f):
        line = re.sub(TINY_EOS, LLAMA3_EOS, line)
        train_data.extend(tokenizer.encode(line))
    # llama3 vocab size is 128256
    train_tokens = np.array(train_data, dtype=np.int32)
    np.save(os.path.join(DATA_DIR, "tiny-train.npy"), train_tokens)
    logging.info(f"Saved tiny-train.npy with {len(train_tokens)} tokens")
    del train_data, train_tokens

    f = open(os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt"), "r")
    valid_data = []
    for line in tqdm(f):
        line = re.sub(TINY_EOS, LLAMA3_EOS, line)
        valid_data.extend(tokenizer.encode(line))
    valid_tokens = np.array(valid_data, dtype=np.int32)
    np.save(os.path.join(DATA_DIR, "tiny-valid.npy"), valid_tokens)
    logging.info(f"Saved tiny-valid.npy with {len(valid_tokens)} tokens")
    del valid_data, valid_tokens
