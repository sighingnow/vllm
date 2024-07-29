"""Benchmark offline inference throughput."""
import argparse
import json
import math
import os
import queue
import random
import signal
import threading
import time
import traceback
import queue
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torch.distributed
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm import envs
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.utils import Counter, FlexibleArgumentParser


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def read_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    requests, responses = [], []
    with open(dataset_path) as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if fixed_output_len is None:
                    responses.append(len(line[1:].split(',')))
            else:
                line = line.replace("\\n", "\n")
                requests.append(line)
    responses = responses + [fixed_output_len
                             ] * (len(requests) - len(responses))
    prompt_token_ids = tokenizer(requests).input_ids
    return [(line, len(token_ids), output_len) for line, token_ids, output_len
            in zip(requests, prompt_token_ids, responses)]


def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    distributed_executor_backend: Optional[str],
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,
    load_format: str = EngineArgs.load_format,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        distributed_executor_backend=distributed_executor_backend,
        load_format=load_format,
    )

    # Add the requests to the engine.
    prompts: List[str] = []
    sampling_params: List[SamplingParams] = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len,
            ))

    start = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return end - start


class RequestIssuer:

    def __init__(self,
                 llm,
                 counter,
                 qps: float = 1.0,
                 max_batch_size: int = None,
                 batch_mode: bool = False):
        self.llm = llm
        self.counter = counter
        self.qps = qps
        self.max_batch_size = max_batch_size
        self.batch_mode = batch_mode
        self.requests: queue.Queue = queue.Queue()
        self.thread = None

    def empty(self) -> bool:
        return self.requests.empty()

    def add_request(self, *args, request_id=None, **kwargs):
        if request_id is None:
            request_id = str(next(self.counter))
        self.requests.put((request_id, args, kwargs))
        return request_id

    def start(self):
        if not self.batch_mode:
            return
        self.thread = threading.Thread(target=self._start,
                                       kwargs={"once": False})
        self.thread.start()

    def issue(self, concurrency: int):
        if self.batch_mode:
            return
        if concurrency >= self.max_batch_size:
            return
        self._start(once=True)

    def _start(self, once: bool = False):
        while not self.requests.empty():
            try:
                request_id, args, kwargs = self.requests.get(block=False)
            except queue.Empty:
                break

            if isinstance(self.llm, queue.Queue):
                self.llm.put((request_id, args, kwargs))
            elif hasattr(self.llm, "add_request"):
                self.llm.add_request(request_id, *args, **kwargs)

            if once:
                break
            else:
                time.sleep(1 / self.qps)

    def stop(self):
        self.requests.queue.clear()
        self.qps = 1000

    def join(self):
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()
        self.thread = None


def print_metrics(
    backend: str,
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    ignore_eos: bool,
    dtype: str,
    enforce_eager: bool,
    kv_cache_dtype: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    speculative_model: str,
    num_speculative_tokens: int,
    speculative_draft_tensor_parallel_size: int,
    use_v2_block_manager: bool,
    qps: float,
    max_batch_size: int,
    batch_mode: bool,
    rope_scaling: Optional[dict],
    gpu_memory_utilization: float,
    load_format: Optional[str],
    num_steps: int,
    start: float,
    end: float,
    request_ids: List[str],
    requests: List[Tuple[Union[str, List[int]], int, int]],
    outputs: List[RequestOutput],
    prompt_durations: List[float],
    decode_durations: List[float],
    verbose: Optional[bool],
):
    input_lengths, output_lengths = [], []
    iterations, accept_rates = [], []
    # Time to first token, time per output token, and time per output response.
    ttfts, tpots, tpors = [], [], []
    decoding_durations, query_durations = [], []
    outputs = [outputs[req] for req in request_ids]
    if verbose:
        print("\n".join([
            f"#{i:3d}/{len(outputs):d}: "
            f"{repr(tokenizer.decode(output[-1].outputs[0].token_ids)) if output else None}, "
            f"{output[-1].outputs[0].token_ids if output else None}"
            for i, output in enumerate(outputs)
        ]))
    for i, ((prompt, input_len, _),
            request_outputs) in enumerate(zip(requests, outputs)):
        if not request_outputs:
            continue
        output = request_outputs[-1]
        output_len = len(output.outputs[0].token_ids)
        decoding_duration = request_outputs[-1].timestamp - request_outputs[0].timestamp
        query_duration = decoding_duration + request_outputs[0].latency
        decoding_durations.append(decoding_duration)
        query_durations.append(query_duration)
        if args.speculative_model:
            # align with the statistical logic of our EAGLE reference implementation
            num_iterations = len(request_outputs) - 1
        else:
            num_iterations = len(request_outputs)
        accept_rate = (output_len - num_iterations) / max(1, num_iterations)
        input_lengths.append(input_len)
        output_lengths.append(output_len)
        iterations.append(len(request_outputs))
        accept_rates.append(accept_rate)
        ttfts.append(request_outputs[0].latency)
        for prev, output in zip(request_outputs, request_outputs[1:]):
            num_tokens = output.token_length - prev.token_length
            tpots.extend(output.latency / max(1, num_tokens)
                         for _ in range(num_tokens))
        tpors.extend([output.latency for output in request_outputs[1:]])

    ttfts = ttfts if ttfts else [0]
    tpots = tpots if tpots else [0]
    tpors = tpors if tpors else [0]
    prompt_durations = prompt_durations if prompt_durations else [0]
    decode_durations = decode_durations if decode_durations else [0]

    for key in dir(envs):
        if key and key.startswith('VLLM'):
            print(f'{key}={getattr(envs, key)}')
    for key, value in os.environ.items():
        if key and key.startswith('CUDA'):
            print(f'{key}={value}')

    # ruff
    print(
        f'Model: {model}\n'
        f'Draft model: {speculative_model}\n'
        f'Rope scaling: {rope_scaling}\n'
        f'Lookahead: {num_speculative_tokens}\n'
        f'TP: {tensor_parallel_size}\n'
        f'Draft TP: {speculative_draft_tensor_parallel_size}\n'
        f'GPU memory utilization: {gpu_memory_utilization:.2f}\n'
        f'Load format: {load_format}\n'
        f'Use v2 block manager: {use_v2_block_manager}\n'
        f'Enforce eager: {enforce_eager}\n'
        f'Quantization: {quantization}\n'
        f'Dtype: {dtype}\n'
        f'KV cache dtype: {kv_cache_dtype}\n'
        f'Random seed: {seed}\n'
        f'N: {n}\n'
        f'Use beam search: {use_beam_search}\n'
        f'Temperature: {temperature:.2f}\n'
        f'top_p: {top_p:.3f}\n'
        f'top_k: {top_k}\n'
        f'Repetition penalty: {repetition_penalty:.3f}\n'
        f'Ignore eos: {ignore_eos}\n'
        f'Enable prefix caching: {enable_prefix_caching}\n'
        f'Enable chunked prefill: {enable_chunked_prefill}\n'
        f'Token budget: {max_num_batched_tokens}\n'
        f'Batch size: {max_batch_size}\n'
        f'Num requests: {len(requests)}/{len(request_ids)}\n'
        f'QPS: {qps}\n'
        f'Batch mode: {batch_mode}\n'
        f'Input tokens: {np.sum(input_lengths)}\n'
        f'Avg input tokens: {np.mean(input_lengths):.2f}\n'
        f'Output tokens: {np.sum(output_lengths)}\n'
        f'Avg output tokens: {np.mean(output_lengths):.2f}\n'
        f'Engine steps: {num_steps}\n'
        f'Accept rate: {np.mean(accept_rates or [0]) :.3f}\n'
        f'Mean TTFT: {np.mean(ttfts) * 1000.0:.2f} ms\n'
        f'Median TTFT: {np.median(ttfts) * 1000.0:.2f} ms\n'
        f'P99 TTFT: {np.percentile(ttfts, 99) * 1000.0:.2f} ms\n'
        f'P95 TTFT: {np.percentile(ttfts, 95) * 1000.0:.2f} ms\n'
        f'Mean TPOT: {np.mean(tpots) * 1000.0:.2f} ms\n'
        f'Median TPOT: {np.median(tpots) * 1000.0:.2f} ms\n'
        f'P99 TPOT: {np.percentile(tpots, 99) * 1000.0:.2f} ms\n'
        f'P95 TPOT: {np.percentile(tpots, 95) * 1000.0:.2f} ms\n'
        f'Mean TPOR: {np.mean(tpors) * 1000.0:.2f} ms\n'
        f'Median TPOR: {np.median(tpors) * 1000.0:.2f} ms\n'
        f'P99 TPOR: {np.percentile(tpors, 99) * 1000.0:.2f} ms\n'
        f'P95 TPOR: {np.percentile(tpors, 95) * 1000.0:.2f} ms\n'
        f'Duration: {end - start:.2f} ({np.sum(prompt_durations):.2f}, {np.sum(decode_durations):.2f}) seconds\n'  # noqa: E501
        f'Throughput: {len(requests) / (end - start):.2f} requests/s\n'  # noqa: E501
        f'Token throughput: {(np.sum(input_lengths) + np.sum(output_lengths)) / (end - start):.2f} tokens/s\n'  # noqa: E501
        f'Output token throughput: {np.sum(output_lengths) / (end - start):.2f} tokens/s\n'  # noqa: E501
        f'Avg query token throughput: {np.mean([output_len / duration if output_len else 0 for output_len, duration in zip(output_lengths, query_durations)]):.2f} tokens/s\n'  # noqa: E501
        f'Avg query generation token throughput: {np.mean([output_len / duration if output_len > 1 else 0 for output_len, duration in zip(output_lengths, decoding_durations)]):.2f} tokens/s\n'  # noqa: E501
        f'Avg query message generation token throughput: {np.mean([output_len / duration for output_len, duration in zip(output_lengths, decoding_durations) if output_len > 1]):.2f} tokens/s\n'  # noqa: E501
        f'Avg step duration: ({np.mean(prompt_durations):.4f}, {np.mean(decode_durations):.4f}) seconds\n'  # noqa: E501
        f'P99 step duration: ({np.percentile(prompt_durations, 99):.4f}, {np.percentile(decode_durations, 99):.4f}) seconds\n'  # noqa: E501
        f'P95 step duration: ({np.percentile(prompt_durations, 95):.4f}, {np.percentile(decode_durations, 95):.4f}) seconds\n'  # noqa: E501
    )
    # yapf: enable


def run_vllm_v2(
    requests: List[Tuple[Union[str, List[int]], int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    ignore_eos: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    distributed_executor_backend: Optional[str],
    num_lookahead_slots: int = 0,
    speculative_model: str = None,
    num_speculative_tokens: int = None,
    speculative_draft_tensor_parallel_size: int = None,
    speculative_max_model_len: int = None,
    use_v2_block_manager: bool = False,
    qps: float = math.inf,
    max_batch_size: int = None,
    batch_mode: bool = False,
    rope_scaling: Optional[dict] = None,
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,
    load_format: Optional[str] = None,
    verbose: Optional[bool] = False,
) -> float:
    from vllm import LLMEngine, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLMEngine.from_engine_args(
        EngineArgs(
            model=model,
            tokenizer=tokenizer,
            skip_tokenizer_init=True,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            kv_cache_dtype=kv_cache_dtype,
            quantization_param_path=quantization_param_path,
            device=device,
            enable_prefix_caching=enable_prefix_caching,
            load_format=load_format,
            download_dir=download_dir,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            distributed_executor_backend=distributed_executor_backend,
            num_lookahead_slots=num_lookahead_slots,
            speculative_model=speculative_model,
            num_speculative_tokens=num_speculative_tokens,
            speculative_draft_tensor_parallel_size=
            speculative_draft_tensor_parallel_size,
            speculative_max_model_len=speculative_max_model_len,
            use_v2_block_manager=use_v2_block_manager,
            max_num_seqs=max_batch_size,
            rope_scaling=rope_scaling,
        ))

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer, trust_remote_code=trust_remote_code)

    requests = [
        (prompt if isinstance(prompt, list) else tokenizer(prompt)["input_ids"],
         input_len, output_len)
        for prompt, input_len, output_len in requests
    ]

    counter = Counter()
    request_issuer = RequestIssuer(llm,
                                   counter,
                                   qps=qps,
                                   max_batch_size=max_batch_size,
                                   batch_mode=batch_mode)

    # Add the requests to the engine.
    request_ids = []
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else temperature,
            top_p=top_p,
            top_k=top_k,
            use_beam_search=use_beam_search,
            ignore_eos=ignore_eos,
            max_tokens=output_len,
            repetition_penalty=repetition_penalty,
        )
        request_ids.append(
            request_issuer.add_request(
                inputs=TokensPrompt(prompt_token_ids=prompt),
                params=sampling_params,
            ))

    num_steps: int = 0
    prompt_durations, decode_durations = [], []
    outputs: Dict[str, List[RequestOutput]] = defaultdict(list)

    request_issuer.start()
    time.sleep(1)
    pbar: tqdm = tqdm(total=len(requests),
                      desc="Processed",
                      dynamic_ncols=True)

    start = time.perf_counter()
    try:
        while (llm.has_unfinished_requests() or
               (pbar.n < pbar.total and not request_issuer.empty())):
            step_start = time.perf_counter()
            step_outputs = llm.step()
            step_finish = time.perf_counter()
            step_duration = step_finish - step_start
            for output in step_outputs:
                output.token_length = len(output.outputs[0].token_ids)
                output.latency = step_duration
                output.timestamp = step_finish
                if output.token_length != 0:  # chunked prefill
                    while (outputs[output.request_id]
                           and outputs[output.request_id][-1].token_length >=
                           output.token_length):
                        outputs[output.request_id].pop(-1)
                    outputs[output.request_id].append(output)
                if output.finished:
                    pbar.update(1)
            if qps > 0 and not step_outputs:
                # Sleep for a while to avoid busy waiting.
                time.sleep(1 / qps)
            prompt_run = all(
                len(output.outputs[0].token_ids) == 1
                for output in step_outputs)
            if prompt_run:
                prompt_durations.append(step_duration)
            else:
                decode_durations.append(step_duration)
            num_steps += 1

            # issue a new request if the current batch is not full
            if not batch_mode:
                request_issuer.issue(
                    concurrency=llm.get_num_unfinished_requests())
    except:  # noqa: E722
        traceback.print_exc()
    finally:
        request_issuer.stop()
        request_issuer.join()
        pbar.close()
    end = time.perf_counter()

    print_metrics(
        backend='vllm',
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        n=n,
        use_beam_search=use_beam_search,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        ignore_eos=ignore_eos,
        dtype=dtype,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        speculative_model=speculative_model,
        num_speculative_tokens=num_speculative_tokens,
        speculative_draft_tensor_parallel_size=speculative_draft_tensor_parallel_size,
        use_v2_block_manager=use_v2_block_manager,
        qps=qps,
        max_batch_size=max_batch_size,
        batch_mode=batch_mode,
        rope_scaling=rope_scaling,
        gpu_memory_utilization=gpu_memory_utilization,
        load_format=load_format,
        num_steps=num_steps,
        start=start,
        end=end,
        request_ids=request_ids,
        requests=requests,
        outputs=outputs,
        prompt_durations=prompt_durations,
        decode_durations=decode_durations,
        verbose=verbose,
    )
    return end - start


def run_sglang(
    requests: List[Tuple[Union[str, List[int]], int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    ignore_eos: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    distributed_executor_backend: Optional[str],
    num_lookahead_slots: int = 0,
    speculative_model: str = None,
    num_speculative_tokens: int = None,
    speculative_draft_tensor_parallel_size: int = None,
    speculative_max_model_len: int = None,
    use_v2_block_manager: bool = False,
    qps: float = math.inf,
    max_batch_size: int = None,
    batch_mode: bool = False,
    rope_scaling: Optional[dict] = None,
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,
    load_format: Optional[str] = None,
    verbose: Optional[bool] = False,
) -> float:
    from sglang.srt.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.managers.io_struct import BatchTokenIDOut, TokenizedGenerateReqInput
    from sglang.srt.managers.controller.tp_worker import broadcast_recv_input, launch_tp_servers, ModelTpServer

    # disable custom all reduce
    from vllm.distributed.parallel_state import set_custom_all_reduce
    set_custom_all_reduce(False)

    nccl_port: int = 28888
    server_args: ServerArgs = ServerArgs(
        # Model and tokenizer
        model_path=model,
        tokenizer_path=tokenizer,
        load_format=load_format,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        context_length=max_model_len,
        quantization=quantization,
        mem_fraction_static=gpu_memory_utilization,
        max_prefill_tokens=max_num_batched_tokens,
        max_running_requests=max_batch_size,
        max_num_reqs=max_batch_size,
        schedule_heuristic="lpm",
        schedule_conservativeness=1.0,
        tp_size=tensor_parallel_size,
        stream_interval=1,
        random_seed=seed,
        log_level="info",
        log_level_http="info",
        log_requests=False,
        show_time_cost=False,
        dp_size=1,
        load_balance_method="round_robin",
        disable_flashinfer=False,
        disable_radix_cache=not enable_prefix_caching,
        disable_regex_jump_forward=True,
        disable_cuda_graph=enforce_eager,
        disable_disk_cache=True,
        enable_torch_compile=False,
        attention_reduce_in_fp32=True,
        enable_p2p_check=False,
        efficient_weight_load=False,
        nccl_init_addr=None,
        nnodes=1,
        node_rank=None,
    )


    if tensor_parallel_size > 1:
        tp_procs = launch_tp_servers(
            gpu_ids=range(0, tensor_parallel_size),
            tp_rank_range=range(1, tensor_parallel_size),
            server_args=server_args,
            nccl_port=nccl_port,
            model_overide_args=None,
        )
    else:
        tp_procs = []
    llm = ModelTpServer(
        gpu_id=0,
        tp_rank=0,
        server_args=server_args,
        nccl_port=nccl_port,
        model_overide_args=None,
    )
    incoming_requests: queue.Queue = queue.Queue()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer, trust_remote_code=trust_remote_code)
    requests = [
        (prompt if isinstance(prompt, list) else tokenizer(prompt)["input_ids"],
         input_len, output_len)
        for prompt, input_len, output_len in requests
    ]

    counter = Counter()
    request_issuer = RequestIssuer(incoming_requests,
                                   counter,
                                   qps=qps,
                                   max_batch_size=max_batch_size,
                                   batch_mode=batch_mode)

    # Add the requests to the engine.
    request_ids = []
    stop_strs = [tokenizer.decode([eos_token_id])
                 for eos_token_id in [tokenizer.eos_token_id, 151643, 151645]]
    for i, (prompt, _, output_len) in enumerate(requests):
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else temperature,
            top_p=top_p,
            top_k=top_k,
            ignore_eos=ignore_eos,
            max_new_tokens=output_len,
            stop=[] if ignore_eos else stop_strs,
        )
        sampling_params.stop_str_max_len = 1
        req = TokenizedGenerateReqInput(
            rid=str(i),
            input_text="",
            input_ids=prompt,
            pixel_values=None,
            image_hash=0,
            image_size=[],
            sampling_params=sampling_params,
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            stream=True,
        )
        request_ids.append(
            request_issuer.add_request(
                request_id=str(i),
                req=req,
            ))

    num_steps: int = 0
    prompt_durations, decode_durations = [], []
    outputs: Dict[str, List[RequestOutput]] = defaultdict(list)

    request_issuer.start()
    time.sleep(1)
    pbar: tqdm = tqdm(total=len(requests),
                      desc="Processed",
                      dynamic_ncols=True)

    start = time.perf_counter()
    try:
        while (not incoming_requests.empty() or
               (llm.running_batch and llm.running_batch.reqs) or
               (pbar.n < pbar.total and not request_issuer.empty())):
            reqs = []
            while not incoming_requests.empty():
                request_id, args, kwargs = incoming_requests.get()
                reqs.append(kwargs['req'])
            if tensor_parallel_size > 1:
                broadcast_recv_input(reqs, 0, llm.model_runner.tp_group.cpu_group)
            step_start = time.perf_counter()
            step_outputs: BatchTokenIDOut = llm.exposed_step(reqs)
            step_finish = time.perf_counter()
            step_duration = step_finish - step_start
            for step_output in step_outputs:
                for (req_id,
                     output_ids,
                     read_offset,
                     finish_reason
                ) in zip(step_output.rids,
                         step_output.decode_ids,
                         step_output.read_offsets,
                         step_output.finished_reason):
                    output: RequestOutput = RequestOutput(
                        request_id=req_id,
                        prompt=None,
                        prompt_token_ids=[],
                        prompt_logprobs=None,
                        outputs=[CompletionOutput(index=0,
                                                  text='',
                                                  token_ids=output_ids[read_offset:],
                                                  cumulative_logprob=0,
                                                  logprobs=None,
                                                  finish_reason=finish_reason)],
                        finished=finish_reason is not None,
                    )
                    output.token_length = len(output.outputs[0].token_ids)
                    output.latency = step_duration
                    output.timestamp = step_finish
                    if output.token_length != 0:  # chunked prefill
                        while (outputs[output.request_id]
                               and outputs[output.request_id][-1].token_length >=
                               output.token_length):
                            outputs[output.request_id].pop(-1)
                        outputs[output.request_id].append(output)
                    if output.finished:
                        pbar.update(1)
            if qps > 0 and not step_outputs:
                # Sleep for a while to avoid busy waiting.
                time.sleep(1 / qps)
            prompt_run = bool(reqs)
            if prompt_run:
                prompt_durations.append(step_duration)
            else:
                decode_durations.append(step_duration)
            num_steps += 1

            # issue a new request if the current batch is not full
            if not batch_mode:
                request_issuer.issue(
                    concurrency=len(llm.running_batch.reqs) if llm.running_batch else 0)
    except:  # noqa: E722
        traceback.print_exc()
    finally:
        request_issuer.stop()
        request_issuer.join()
        pbar.close()
    end = time.perf_counter()

    for proc in tp_procs:
        os.kill(proc.pid, signal.SIGKILL)

    print_metrics(
        backend='sglang',
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        n=n,
        use_beam_search=use_beam_search,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        ignore_eos=ignore_eos,
        dtype=dtype,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        speculative_model=speculative_model,
        num_speculative_tokens=num_speculative_tokens,
        speculative_draft_tensor_parallel_size=speculative_draft_tensor_parallel_size,
        use_v2_block_manager=use_v2_block_manager,
        qps=qps,
        max_batch_size=max_batch_size,
        batch_mode=batch_mode,
        rope_scaling=rope_scaling,
        gpu_memory_utilization=gpu_memory_utilization,
        load_format=load_format,
        num_steps=num_steps,
        start=start,
        end=end,
        request_ids=request_ids,
        requests=requests,
        outputs=outputs,
        prompt_durations=prompt_durations,
        decode_durations=decode_durations,
        verbose=verbose,
    )
    return end - start


def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    use_beam_search: bool,
    max_batch_size: int,
    trust_remote_code: bool,
) -> float:
    assert not use_beam_search
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (max(max_prompt_len, next_prompt_len) +
                    max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt",
                              padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_mii(
    requests: List[Tuple[str, int, int]],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
) -> float:
    from mii import client, serve
    llm = serve(model, tensor_parallel=tensor_parallel_size)
    prompts = [prompt for prompt, _, _ in requests]

    start = time.perf_counter()
    llm.generate(prompts, max_new_tokens=output_len)
    end = time.perf_counter()
    client = client(model)
    client.terminate_server()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.dataset is None:
        if args.num_prompts is None:
            args.num_prompts = 1000
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_prompts)]
    else:
        if args.dataset.endswith("jsonl"):
            if args.num_prompts is None:
                args.num_prompts = 1000
            requests = sample_requests(args.dataset, args.num_prompts,
                                       tokenizer, args.output_len)
        else:
            if args.output_len is None:
                args.output_len = 2048
            requests = read_requests(args.dataset, tokenizer, args.output_len)
            if args.num_prompts is not None:
                requests = requests[:args.num_prompts]

    if args.backend == "vllm":
        elapsed_time = run_vllm_v2(
            requests, args.model, args.tokenizer, args.quantization,
            args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
            args.temperature, args.top_p, args.top_k, args.repetition_penalty,
            args.ignore_eos, args.trust_remote_code, args.dtype,
            args.max_model_len, args.enforce_eager, args.kv_cache_dtype,
            args.quantization_param_path, args.device,
            args.enable_prefix_caching, args.enable_chunked_prefill,
            args.max_num_batched_tokens, args.distributed_executor_backend,
            args.num_lookahead_slots, args.speculative_model,
            args.num_speculative_tokens,
            args.speculative_draft_tensor_parallel_size,
            args.speculative_max_model_len, args.use_v2_block_manager,
            args.qps, args.hf_max_batch_size, args.batch_mode,
            args.rope_scaling, args.gpu_memory_utilization, args.download_dir,
            args.load_format, args.verbose)
    elif args.backend == "sglang":
        elapsed_time = run_sglang(
            requests, args.model, args.tokenizer, args.quantization,
            args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
            args.temperature, args.top_p, args.top_k, args.repetition_penalty,
            args.ignore_eos, args.trust_remote_code, args.dtype,
            args.max_model_len, args.enforce_eager, args.kv_cache_dtype,
            args.quantization_param_path, args.device,
            args.enable_prefix_caching, args.enable_chunked_prefill,
            args.max_num_batched_tokens, args.distributed_executor_backend,
            args.num_lookahead_slots, args.speculative_model,
            args.num_speculative_tokens,
            args.speculative_draft_tensor_parallel_size,
            args.speculative_max_model_len, args.use_v2_block_manager,
            args.qps, args.hf_max_batch_size, args.batch_mode,
            args.rope_scaling, args.gpu_memory_utilization, args.download_dir,
            args.load_format, args.verbose)
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size,
                              args.trust_remote_code)
    elif args.backend == "mii":
        elapsed_time = run_mii(requests, args.model, args.tensor_parallel_size,
                               args.output_len)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)

    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        os.kill(child.pid, signal.SIGKILL)

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii", "sglang"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--ignore-eos", action="store_true", default=False)
    parser.add_argument("--num-prompts",
                        type=int,
                        default=None,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
        'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "openvino", "tpu", "xpu"],
        help='device type for vLLM execution, supporting CUDA, OpenVINO and '
        'CPU.')
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="enable automatic prefix caching for vLLM backend.")
    parser.add_argument("--enable-chunked-prefill",
                        action='store_true',
                        help="enable chunked prefill for vLLM backend.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--num-lookahead-slots',
                        type=int,
                        default=0,
                        help='Experimental scheduling config necessary for '
                        'speculative decoding. This will be replaced by '
                        'speculative config in the future; it is present '
                        'to enable correctness tests until then.')
    parser.add_argument(
        '--speculative-model',
        type=str,
        default=None,
        help='The name of the draft model to be used in speculative decoding.')
    parser.add_argument('--num-speculative-tokens',
                        type=int,
                        default=None,
                        help='The number of speculative tokens to sample from '
                        'the draft model in speculative decoding.')
    parser.add_argument('--speculative-draft-tensor-parallel-size',
                        type=int,
                        default=None,
                        help='Number of tensor parallel replicas for '
                        'the draft model in speculative decoding.')
    parser.add_argument('--speculative-max-model-len',
                        type=int,
                        default=None,
                        help='The maximum sequence length supported by the '
                        'draft model. Sequences over this length will skip '
                        'speculation.')
    parser.add_argument('--use-v2-block-manager',
                        action='store_true',
                        help='Use BlockSpaceMangerV2.')
    parser.add_argument("--qps",
                        type=float,
                        default='inf',
                        help="Queries per second for the benchmark.")
    parser.add_argument("--batch-mode",
                        action="store_true",
                        help="Use batch mode, i.e., enable batching prefills.")
    parser.add_argument('--rope-scaling',
                        default=None,
                        type=json.loads,
                        help='RoPE scaling configuration in JSON format. '
                        'For example, {"type":"dynamic","factor":2.0}')
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument(
        '--distributed-executor-backend',
        choices=['ray', 'mp'],
        default=None,
        help='Backend to use for distributed serving. When more than 1 GPU '
        'is used, will be automatically set to "ray" if installed '
        'or "mp" (multiprocessing) otherwise.')
    parser.add_argument(
        '--load-format',
        type=str,
        default=EngineArgs.load_format,
        choices=[
            'auto', 'pt', 'safetensors', 'npcache', 'dummy', 'tensorizer',
            'bitsandbytes'
        ],
        help='The format of the model weights to load.\n\n'
        '* "auto" will try to load the weights in the safetensors format '
        'and fall back to the pytorch bin format if safetensors format '
        'is not available.\n'
        '* "pt" will load the weights in the pytorch bin format.\n'
        '* "safetensors" will load the weights in the safetensors format.\n'
        '* "npcache" will load the weights in pytorch format and store '
        'a numpy cache to speed up the loading.\n'
        '* "dummy" will initialize the weights with random values, '
        'which is mainly for profiling.\n'
        '* "tensorizer" will load the weights using tensorizer from '
        'CoreWeave. See the Tensorize vLLM Model script in the Examples'
        'section for more information.\n'
        '* "bitsandbytes" will load the weights using bitsandbytes '
        'quantization.\n')
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    if (not args.speculative_model or
        (not args.num_speculative_tokens or args.num_speculative_tokens <= 0)):
        args.speculative_model = None
        args.num_speculative_tokens = None

    if args.backend in ["vllm", "sglang"]:
        if args.hf_max_batch_size is None:
            args.hf_max_batch_size = 16
        # if args.hf_max_batch_size is not None:
        #     raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
    main(args)
