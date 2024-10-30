import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List, Union
import os
from os.path import join as pjoin

import utils
import memory as mem_
from llama_lora.generation import Llama, sample_top_p
from llama_lora.model import forward_no_embeddings
from bookcorpus import dataset_bookcorpus as ds_bc

from dataclasses import dataclass


@dataclass
class MemoryLLamaTrainArgs:
    train_name: str = "Train"
    llama_version = "llama3.2-1B"
    llama_ckpt_dir: str = "./.llama/checkpoints/Llama3.2-1B-Instruct"
    llama_tokenizer_path: str = "./.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model"
    memory_ckpt_dir: str = "./checkpoints"
    max_seq_len: int = 1024
    max_batch_size: int = 32
    mem_cycle_len: int = 32
    max_gen_len: int = 128
    save: bool = True
    save_every: int = 5

def main():
    batch_per_step = 8
    save_llama = False
    
    logger_args = utils.LoggerArgs()
    train_args = MemoryLLamaTrainArgs(mem_cycle_len=16,max_gen_len=8)
    memory_args = mem_.MemoryArgs(summary_len=4, stride=4)

    logger = utils.setup_logger(logger_args)
    chrono = utils.Chrono()

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    logger.info("Using cuda:0")

    x = ["Today is Monday, Tommorow is Tuesday,", "Hellow, this is president speaking. ", "What day is "]
    # Building Llama
    logger.info("Building Llama")
    llama = Llama.build(
        train_args.llama_ckpt_dir,
        train_args.llama_tokenizer_path,
        train_args.max_seq_len,
        train_args.max_batch_size,
    )
    try:
        llama_pt = pjoin(train_args.memory_ckpt_dir, "llama.pt")
        llama.model.load_state_dict(torch.load(llama_pt, map_location="cpu"))
        logger.info("Found a chekpoint of Llama at: " + llama_pt)
    except FileNotFoundError:
        logger.info("Found no checkpoint of Llama.")
    llama.model.requires_grad_(False)
    llama.model.to(device)

    # Loading Memory
    memory = mem_.MultiHeadMemory(
        train_args.max_seq_len,
        8,
        memory_args.dim,
        memory_args.memory_size,
        memory_args.hdim,
        memory_args.summary_len,
        memory_args.stride,
    )
    try:
        memory_pt = pjoin(train_args.memory_ckpt_dir, "memory.pt")
        memory.load_state_dict(torch.load(memory_pt, map_location="cpu"))
        logger.info("Found a chekpoint of memory at: " + memory_pt)
    except FileNotFoundError:
        logger.info("Found no checkpoint of memory.")
    memory.to(device)

    memory.eval()
    t = [llama.tokenizer.encode(s, bos=True, eos=False) for s in x]
    with chrono.measure("fprop"):
        logit, gt_logit = generate_logits(
            llama, memory, train_args.mem_cycle_len, t, train_args.max_gen_len
        )
    print(gt_logit.shape)
    gt_t = [llama.tokenizer.decode(gt_) for gt_ in gt_logit.argmax(dim=1).detach().cpu().numpy()]
    generated_t = [llama.tokenizer.decode(generated_logit) for generated_logit in logit.argmax(dim=1).detach().cpu().numpy()]
    print(generated_t, gt_t)


def generate_logits(
    llama,
    memory,
    mem_cycle_len,
    batch_tokens: List[List[int]],
    max_gen_len: int,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> Union[torch.Tensor, torch.Tensor]:
    """
    Returns (logits, gt_logits).
    Shape of each logit is (batch_size, n_vocabs, seqlen) for CrossEntropyLoss.
    To calculate loss, do criterion(*generate_logits(...)) or
      criterion(input=generated_logits[0], target=generated_logits[1]).
    This does not call zero_grad() for llama.model and memory.
    """

    # Initializing local variables
    bsz = len(batch_tokens)
    params = llama.model.params
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    min_prompt_len = min(len(t) for t in batch_tokens)
    max_prompt_len = max(len(t) for t in batch_tokens)
    assert max_prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    losses = []
    arr_gt_logits = []
    arr_logits = []

    mem_len = memory.memory_len()
    h_mem = memory.init_hidden(bsz)

    # initializing kv cache
    llama.model.init_cache()

    # padding tokens
    pad_id = llama.tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(batch_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    # preparation for llama.generate
    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device="cuda")
    input_text_mask = tokens != pad_id

    # initial h_mem
    mem_cycles = min_prompt_len // mem_cycle_len
    mem_start_pos = mem_cycles * mem_cycle_len
    mem_pos = min_prompt_len % mem_cycle_len
    for i in range(mem_cycles):
        h_mem = memory.forward(
            llama.model.tok_embeddings(
                tokens[:, i * mem_cycle_len : (i + 1) * mem_cycle_len]
            ),
            h_mem,
        )

    # if min_prompt_len == total
    if min_prompt_len == total_len:
        gt_logit = llama.model.forward(tokens, prev_pos)[:, -1]
        logit = forward_no_embeddings(
            llama.model,
            torch.cat(
                (
                    memory.read(h_mem),
                    llama.model.tok_embeddings(tokens[:, mem_start_pos:]),
                ),
                dim=1,
            ),
            prev_pos,
        )[:, -1]
        arr_gt_logits.append(gt_logit)
        arr_logits.append(logit)

    # generating gt logits
    num_generations = total_len - min_prompt_len
    with torch.no_grad():
        stop_tokens = torch.tensor(list(llama.tokenizer.stop_tokens))
        for cur_pos in range(min_prompt_len, total_len):
            gt_logits = llama.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            arr_gt_logits.append(gt_logits[:, -1])

            if temperature > 0:
                probs = torch.softmax(gt_logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(gt_logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)

            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            # checking eos reached
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                num_generations = cur_pos - min_prompt_len + 1
                break

    # generate logits:
    prev_pos = 0
    llama.model.clear_cache()
    for cur_pos in range(
        mem_start_pos + mem_pos, mem_start_pos + mem_pos + num_generations
    ):
        mem_logits = forward_no_embeddings(
            llama.model,
            (
                torch.cat(
                    (
                        memory.read(h_mem),
                        llama.model.tok_embeddings(tokens[:, mem_start_pos:cur_pos]),
                    ),
                    dim=1,
                )
                if mem_start_pos == prev_pos or prev_pos == 0
                else llama.model.tok_embeddings(tokens[:, prev_pos:cur_pos])
            ),
            (
                0
                if prev_pos % mem_cycle_len == 0
                else mem_len + (prev_pos % mem_cycle_len)
            ),
        )
        arr_logits.append(mem_logits[:, -1])
        prev_pos = cur_pos

        # update memory
        if cur_pos - mem_start_pos == mem_cycle_len:
            h_mem = memory.forward(
                llama.model.tok_embeddings(
                    tokens[:, mem_start_pos : cur_pos]
                ),
                h_mem,
            )
            mem_start_pos = mem_start_pos + mem_cycle_len

    return torch.stack(arr_logits).permute((1, 2, 0)), torch.stack(
        arr_gt_logits
    ).permute((1, 2, 0))


if __name__ == "__main__":
    os.environ["HF_DATASETS_CACHE"] = "./bookcorpus/cache"
    main()
