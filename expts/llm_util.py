"""
file: llm_util.py
----------------
Utility functions for LLM experiment
"""

import os
import re
import argparse
import logging
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
from llm import TransformerLM

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


def chatbot(model_name):
    tokenizer = AutoTokenizer.from_pretrained("llama3-8b")
    LLAMA3_EOS = "<|end_of_text|>"

    model = TransformerLM(
        vocab_size=128256,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=2048,
        attn_pdrop=0.1,
        residual_pdrop=0.1,
    )
    model.load_state_dict(torch.load(f"expts/model/{model_name}.pth"))
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
    else:
        device = torch.device("cpu")
    basename = os.path.basename(model_name)
    print("=" * 80)
    print(f"Hi! I'm ChatBob (base:{basename}). Type 'exit' to quit.")
    print("=" * 80)
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        user_input = user_input + LLAMA3_EOS
        user_input = tokenizer.encode(user_input)
        user_input = torch.tensor(user_input).unsqueeze(0)
        user_input = user_input.to(device)
        output, token_probs = model.generate(
            user_input, max_new_tokens=256, temperature=0.7, top_k=10
        )
        perplexity = np.exp(-np.mean(token_probs))
        output_str = ""
        for token in output[0]:
            if token.item() == tokenizer.bos_token_id:
                continue
            if token.item() == tokenizer.eos_token_id:
                break
            output_str += tokenizer.decode(token.item())
        print("-" * 80)
        print(f"ChatBob: {output_str}")
        print(f"Response Perplexity: {perplexity}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llm-0")
    args = parser.parse_args()

    chatbot(args.model)
