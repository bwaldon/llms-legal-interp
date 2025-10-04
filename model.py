from typing import List, Tuple
from collections import OrderedDict
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, CompletionOutput
import torch

from prompts import candidates

class MetaLinguisticJudgement:
    def __init__(self, model_name, seed, max_model_len=216):
        self.model_name = model_name
        self.infer_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=64,
            seed=seed)
        self.logprob_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=1,
            seed=seed,
            logprobs=1,
            prompt_logprobs=True
        )

        if 'gpt' in model_name:
            max_model_len = 256
        elif 'bloom' in model_name:
            max_model_len = 224
        elif 'ministral' in model_name:
            max_model_len = 240
        if 'bloom' in model_name:
            gpu_memory_utilization = 0.3
        else:
            gpu_memory_utilization = 0.92
        self.max_model_len = max_model_len

        if "70B" in self.model_name:
            tensor_parallel_size = 4
        else:
            tensor_parallel_size = 1

        num_gpus = torch.cuda.device_count()
        if num_gpus < tensor_parallel_size:
            raise RuntimeError(
                    f"Requested tensor_parallel_size={tp_size}, "
                            f"but only {num_gpus} GPU(s) are available."
                                )
        self.llm = LLM(
            model_name,
            max_model_len=max_model_len,
            seed=seed,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=1,
            tensor_parallel_size=tensor_parallel_size,
            disable_custom_all_reduce=True
        )

    def infer(self, prompts: List[str]) -> List[CompletionOutput]:
        outputs = self.llm.generate(prompts, self.infer_params)
        return [output.outputs[0] for output in outputs]

    def probs(self, prompts: List[str]) -> dict[str, List[float]]:
        # BLOOM has issues with sentence # 903, 1040
        self.llm.llm_engine.max_num_seqs = 1

        SPACE= " "
        # We use lowercase yes and no in the prompts now.
        # candidates = ['YES', 'Yes', 'yes', 'NO', 'No', 'no', 'A', 'B']

        prompts_with_token = [p + SPACE + answer for p in prompts for answer in candidates]
        outputs = self.llm.generate(prompts_with_token, self.logprob_params)
        # TODO implement sequence probability the same way
        tokenizer = self.llm.get_tokenizer()
        #TODO: could be min float
        MIN_INT = np.iinfo(int).min
        candidate_tokens = OrderedDict()
        for candidate in candidates:
            candidate_tokens[candidate] = tokenizer(SPACE + candidate)["input_ids"][-1]

        # Initialize Dict of Lists (Columnar)
        candidate_logprobs = OrderedDict()
        for candidate in candidates:
            candidate_logprobs[candidate] = list()

        for batch_start in range(0, len(outputs), len(candidates)):
            prompt_batch = outputs[batch_start: batch_start + len(candidates)]
            for candidate, output in zip(candidates, prompt_batch):
                token_id = candidate_tokens[candidate]
                candidate_logprobs[candidate].append(
                    output.prompt_logprobs[-1][token_id].logprob if output.prompt_logprobs is not None else MIN_INT
                )

        return candidate_logprobs
