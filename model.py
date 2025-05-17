from prompts import MetaLinguisticPrompt
from vllm import LLM, SamplingParams, CompletionOutput
from typing import List


class MetaLinguisticJudgement:
    def __init__(self, model_name, seed, max_model_len=256):
        self.model_name = model_name
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256, seed=seed, logprobs=16)
        self.max_model_len = max_model_len
        self.llm = LLM(
            model_name,
            max_model_len=max_model_len,
            seed=seed,
            dtype="float16",
            gpu_memory_utilization=0.95,
            max_num_seqs=8,
            swap_space=8,
            max_num_batched_tokens=256
        )

    def infer(self, prompts: List[MetaLinguisticPrompt]) -> List[CompletionOutput]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0] for output in outputs]

    def probs(self, prompts: List[MetaLinguisticPrompt]) -> List:
        outputs = self.llm.generate([p.text for p in prompts], self.sampling_params)
        return [output.outputs[0].logprobs for output in outputs]
