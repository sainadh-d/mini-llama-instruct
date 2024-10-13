import os
import time
import math
import torch
import numpy as np
import argparse
import traceback
from tokenizer import Tokenizer
from model import Transformer, MiniLlamaArgs
from torch.nn import functional as F


# ---------- Data Loader ---------------------
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        global master_process
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = data_folder
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# -------------- Distributed Data Parallel ---------------------

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
master_process = True
ddp = None

def preprocess_function(example):
    """
    Formatting function for Instruct training.
    """
    text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return text

if __name__ == "__main__":
    # Get Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="", help="the data folder to use for training")
    parser.add_argument("--log-dir", type=str, default=False, required=True, help="Log directory to use")
    args = parser.parse_args()

    data_folder = args.data
    log_dir = args.log_dir

    try:

        # set up DDP (distributed data parallel).
        # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
        ddp = int(os.environ.get("RANK", -1)) != -1 # Is this a ddp run
        if ddp:
            assert torch.cuda.is_available(), "We need CUDA for ddp"
            init_process_group(backend="nccl")
            ddp_rank = int(os.environ['RANK'])
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            ddp_world_size = int(os.environ['WORLD_SIZE'])

            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
            master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        else:
            ddp_rank = 0
            ddp_local_rank = 0
            ddp_world_size = 1
            master_process = True
            # Auto detect device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

        print("Didn't crash yay!")
        device_type = "cuda" if device.startswith("cuda") else "cpu"

        torch.manual_seed(1337)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)

        enc = Tokenizer()

        # --------- Training Llama ----------------------------

        total_batch_size = 524288  # 2**19

        B = 4  # Micro Batch size
        T = 1024  # Sequence length

        assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be a multiple of B * T * ddp_world_size"
        grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
        if master_process:
            print("Total desired batch size: ", total_batch_size)
            print("Grad accumulation steps: ", grad_accum_steps)

        print(f"I am process: {ddp_rank}, using device: {device} and the world_size is {ddp_world_size}")

        train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
        val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

        model = Transformer(MiniLlamaArgs())

        print(f"Loading checkpoint: tiny_llama.pt")
        checkpoint_file = "tiny_llama.pt"
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        weights = checkpoint['model']
        model.load_state_dict(weights)

        # Del unnecessary variabels
        del weights
        del checkpoint

        model.to(device)

        # model = torch.compile(model)
        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])

        raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

        max_lr = 6e-4
        min_lr = max_lr * 0.1
        warmup_steps = 3
        max_steps = 8 * 10 # 4e6 (4 million tokens) / 2**19 (tokens per batch) * 10 epochs

        # Optimize!
        optimizer = raw_model.configure_optimizers(
            weight_decay=0.1, learning_rate=max_lr, betas=(0.9, 0.95), device_type=device_type
        )

        # create the log directory we will write checkpoints to and log to
        log_file = os.path.join(log_dir, f"log.txt")
        eval_file = os.path.join(log_dir, f"eval.txt")
        if master_process:
            os.makedirs(log_dir, exist_ok=True)
            with open(log_file, "w") as f: # open for writing to clear the file
                pass
            with open(eval_file, "w") as f: # open for writing to clear the file
                pass

        for step in range(max_steps):
            t0 = time.time()
            last_step = (step == max_steps - 1)

            # once in a while evaluate our validation loss
            if step % 2 == 0 or last_step:
                model.eval()
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = 20
                    for _ in range(val_loss_steps):
                        x, y = val_loader.next_batch()
                        x, y = x.to(device), y.to(device)
                        logits, loss = model(x, y)
                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()
                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    print(f"validation loss: {val_loss_accum.item():.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                    if step > 0 and (step % 500 == 0 or last_step):
                        # optionally write model checkpoints
                        checkpoint_path = os.path.join(log_dir, f"finetune_{step:05d}.pt")
                        checkpoint = {
                            'model': raw_model.state_dict(),
                        }
                        torch.save(checkpoint, checkpoint_path)

            # once in a while generate from the model (except step 0, which is noise)
            if master_process and ((step >= 0 and step % 5 == 0) or last_step):
                model.eval()
                num_return_sequences = 4
                max_length = 32
                prompt = preprocess_function({"instruction": "Add the below numbers", "input": "2, 3", "output": ""})
                tokens = enc.encode(prompt, True, False)
                tokens = torch.tensor(tokens, dtype=torch.long)
                tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
                xgen = tokens.to(device)
                sample_rng = torch.Generator(device=device)
                sample_rng.manual_seed(42 + ddp_rank)
                while xgen.size(1) < max_length:
                    # forward the model to get the logits
                    with torch.no_grad():
                        logits, loss = model(xgen) # (B, T, vocab_size)
                        # take the logits at the last position
                        logits = logits[:, -1, :] # (B, vocab_size)
                        # get the probabilities
                        probs = F.softmax(logits, dim=-1)
                        # do top-k sampling of 50 (huggingface pipeline default)
                        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                        # select a token from the top-k probabilities
                        # note: multinomial does not demand the input to sum to 1
                        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                        # gather the corresponding indices
                        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                        # append to the sequence
                        xgen = torch.cat((xgen, xcol), dim=1)
                # print the generated text
                for i in range(num_return_sequences):
                    tokens = xgen[i, :max_length].tolist()
                    decoded = enc.decode(tokens)
                    print(f"rank {ddp_rank} sample {i}: {decoded}")
                    with open(eval_file, "a") as f:
                        f.write(f"rank {ddp_rank} sample {i}: {decoded}\n")

            model.train()

            loss_accum = 0.0
            # Repeated training and Gradient Accumulation
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                if ddp:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

            if ddp:
                # Get averaged loss across all processes
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Learning rate schedule
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) * 1000  # time diff in ms
            tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
            tps = tokens_processed / (t1 - t0) 
            if master_process:
                print(f"Step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tokens-per-sec: {tps}")
                with open(log_file, "a") as f:
                    f.write(f"{step} train {loss_accum.item():.6f}\n")
    except Exception as e:
        print(f"Exception: {e}")
        print(traceback.format_exc())
    finally:
        if ddp:
            destroy_process_group()
