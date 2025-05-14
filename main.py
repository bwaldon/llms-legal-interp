import argparse
import csv
import gc
from collections import namedtuple

from huggingface_hub import auth_check
import torch

from model import MetaLinguisticPrompt, MetaLinguisticJudgement

import random
import numpy as np

from enum import Enum
from prompts import get_yes_or_no_vague_contracts

def print_logprobs(logprobs):
    for prompt_logprobs in logprobs:
        print(type(prompt_logprobs), prompt_logprobs)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

model_list = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "bigscience/bloom-560m"
    "bigscience/bloom-1b1",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1"
]


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def get_prompts():
    return [
        MetaLinguisticPrompt(topic="landscaping", features=["question", "bool"]),
        MetaLinguisticPrompt(topic="landscaping", features=["question", "bool", "reverse"]),
        MetaLinguisticPrompt(topic="landscaping", features=["arg", "q_agree"]),
        MetaLinguisticPrompt(topic="landscaping", features=["arg", "q_agree", "neg"]),
        MetaLinguisticPrompt(topic="landscaping", features=["neg_arg", "q_agree"]),
        MetaLinguisticPrompt(topic="landscaping", features=["neg_arg", "q_agree", "neg"]),
        MetaLinguisticPrompt(topic="landscaping", features=["event", "coverage", "q_cover"]),
        MetaLinguisticPrompt(topic="landscaping", features=["event", "coverage", "neg_q_cover"]),
        MetaLinguisticPrompt(topic="landscaping", features=["event", "coverage", "judgement", "bool"]),
        MetaLinguisticPrompt(topic="landscaping", features=["event", "coverage", "judgement", "bool", "reverse"]),
        MetaLinguisticPrompt(topic="landscaping", features=["event", "coverage", "judgement", "q_agree"]),
        MetaLinguisticPrompt(topic="landscaping", features=["event", "coverage", "judgement", "q_agree", "neg"]),
        MetaLinguisticPrompt(topic="landscaping", features=["event", "coverage", "judgement", "mc_prompt"]),
        MetaLinguisticPrompt(topic="landscaping", features=["event", "coverage", "judgement", "mc_prompt_reverse"])
    ]
"""
    Interactive snippet for inference
    prompt = MetaLinguisticPrompt(topic="landscaping", features=["question", "bool"])
    model = MetaLinguisticJudgement("meta-llama/Llama-3.2-1B", 42)
    model.infer([prompt])
    
"""
def load_and_infer_with_model(model_name, seed, output_type, prompts, dataset):
    print(model_name + "\n")
    model = MetaLinguisticJudgement(model_name, seed)
    # outputs = model.infer(dataset["prompt"])



    def print_prompts_and_output(p, o):
        result = result_from_output(p, o)
        star_20 = '*' * 20
        output_string = f"""
        {star_20}{result.topic}{star_20}{result.nudge}{star_20}
        
        {p.text}
        
        {result.output}
        
        {result.output_prob}
        """

        # output_string += f"{star_20}{p.topic}{star_20}{p.nudge}{star_20}"
        # output_string += "\n\n"
        # output_string += p.text
        # output_string += "\n\n"
        # output_string += o
        # output_string += "\n\n\n"
        print(output_string)


    def result_from_output(prompt, output):
        PromptResult = namedtuple('PromptResult', ['prompt', 'topic', 'nudge', 'output', 'output_prob', 'output_token_probs', 'token_probs'])
        return PromptResult(prompt, prompt.topic, prompt.nudge , output.text, output.cumulative_logprob,  output.logprobs, None)

    match output_type:
        case OutputType.TEXT:
            outputs = model.infer([prompt.text for prompt in prompts])
            for p, o in zip(prompts, outputs):
                print_prompts_and_output(p, o)
        case OutputType.LOGPROBS:
            outputs = model.probs(prompts)
            print_logprobs(outputs)
    del model
    cleanup()

class OutputType(Enum):
    TEXT = "text"
    LOGPROBS = "logprobs"

def hf_auth_check(model_list):
    for model in model_list:
        auth_check(model)

def main(seed, output_type):
    prompts = get_prompts()
    prompts_dataset = get_yes_or_no_vague_contracts()["test"]
    for model_name in model_list:
        load_and_infer_with_model(model_name, seed, output_type, prompts, prompts_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--type", type=OutputType, default=OutputType.TEXT)
    args = parser.parse_args()
    main(args.seed, args.type)
