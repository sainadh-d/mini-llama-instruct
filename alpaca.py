import os
import multiprocessing as mp
import numpy as np
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
from tokenizer import Tokenizer


# ------------------------------------------
local_dir = "alpaca"
shard_size = int(1e6) # 1M tokens per shard
dataset_name = "tatsu-lab/alpaca"
short_name = "alpaca"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
ds = load_dataset(dataset_name, split="train")

# init the tokenizer
enc = Tokenizer()

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def preprocess_function(example):
    """
    Formatting function returning a list of samples (kind of necessary for SFT API).
    """
    text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return text

def tokenize(example):
    """tokenizes a single document and returns a numpy array of uint16 tokens."""

    # Preprocess the example and convert it to instruction format.
    example = preprocess_function(example)
    tokens = enc.encode(example, True, True) # Use BOS, EOS
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

shard_index = 0
# preallocate buffer to hold current shard
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None

for tokens in map(tokenize, ds):
    # is there enough space in the current shard for the new tokens?
    if token_count + len(tokens) < shard_size:
        # simply append tokens to current shard
        all_tokens_np[token_count:token_count+len(tokens)] = tokens
        token_count += len(tokens)
        # update progress bar
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
    else:
        # write the current shard and start a new one
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{short_name}_{split}_{shard_index:06d}")
        # split the document into whatever fits in this shard; the remainder goes to next one
        remainder = shard_size - token_count
        progress_bar.update(remainder)
        all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        progress_bar = None
        # populate the next shard with the leftovers of the current doc
        all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
        token_count = len(tokens)-remainder

# write any remaining tokens as the last shard
if token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"{short_name}_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])
