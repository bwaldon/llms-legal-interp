from prompts import MetaLinguisticPrompt
from vllm import LLM, SamplingParams, CompletionOutput
from typing import List, Tuple
from transformers import AutoTokenizer


class MetaLinguisticJudgement:
    def __init__(self, model_name, seed, max_model_len=256):
        self.model_name = model_name
        self.infer_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=64,
            seed=seed)
        self.logprob_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1,
            seed=seed,
            logprobs=1,
            prompt_logprobs=True
        )
        self.max_model_len = max_model_len
        self.llm = LLM(
            model_name,
            max_model_len=max_model_len,
            seed=seed,
            dtype="float16",
            gpu_memory_utilization=0.85,
            max_num_seqs=8,
            swap_space=2,
        )

    def infer(self, prompts: List[str]) -> List[CompletionOutput]:
        outputs = self.llm.generate(prompts, self.infer_params)
        return [output.outputs[0] for output in outputs]

    def probs(self, prompts: List[str]) -> Tuple[List, List, List, List]:
        prompts_with_token = []
        for p in prompts:
            prompts_with_token += [p + ' Yes', p + ' No', p + " A", p + " B"]
        self.llm.llm_engine.max_num_seqs = 1
        outputs = self.llm.generate(prompts_with_token, self.logprob_params)

        tokenizer = self.llm.get_tokenizer()
        yes_token_id = tokenizer(" Yes")["input_ids"][-1]
        no_token_id = tokenizer(" No")["input_ids"][-1]
        a_token_id = tokenizer(" A")["input_ids"][-1]
        b_token_id = tokenizer(" B")["input_ids"][-1]

        yes_logprobs = []
        no_logprobs = []
        a_logprobs = []
        b_logprobs = []
        for n, output in enumerate(outputs):
            if n % 4 == 0:
                yes_logprobs.append(output.prompt_logprobs[-1][yes_token_id].logprob)
            elif n % 4 == 1:
                no_logprobs.append(output.prompt_logprobs[-1][no_token_id].logprob)
            elif n % 4 == 2:
                a_logprobs.append(output.prompt_logprobs[-1][a_token_id].logprob)
            elif n % 4 == 3:
                b_logprobs.append(output.prompt_logprobs[-1][b_token_id].logprob)
        return yes_logprobs, no_logprobs, a_logprobs, b_logprobs
