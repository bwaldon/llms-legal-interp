import argparse

import gc
import string
import token
from collections import namedtuple
import re
import random
import numpy as np

from tqdm import tqdm

from huggingface_hub import auth_check
import torch

from prompts import get_dataset_for_coverage_questions
from model import MetaLinguisticPrompt, MetaLinguisticJudgement
from datasets import Dataset



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
    "gpt2-large"
    "gpt2-xl",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-7B-Instruct",
    "mistralai/Ministral-8B-Instruct-2410",
    "google/gemma-7b-it"
]

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
# Interactive snippet for inference
prompt = get_dataset_for_coverage_questions()[0]['prompt']
model = MetaLinguisticJudgement("meta-llama/Llama-3.1-8B", 42)
output = model.infer([prompt])
yes_logprobs, no_logprobs, a_logprobs, b_logprobs = model.probs([prompt])
"""


def load_and_infer_with_model(model_name, seed, dataset):
    def model_cleanup():
        torch.cuda.empty_cache()
        gc.collect()

    print(model_name + "\n")

    model = MetaLinguisticJudgement(model_name, seed)

    def extract_first_answer_token(text):
        return [x.strip("\"'\.!") for x in re.split(r"([a-zA-z]+)?\s+", text) if x][0]

    prompts = dataset["prompt"]
    # outputs = model.infer(prompts)
    model_cleanup()
    yes_logprobs, no_logprobs, a_logprobs, b_logprobs = model.probs(prompts)

    results_dict = {
        "title": dataset["title"],
        "prompt_type": dataset["prompt_type"],
        "prompt": dataset["prompt"],
        "version": dataset["version"],
        "output": [extract_first_answer_token(output.text) for output in outputs],
        "output_text": [output.text for output in outputs],
        "Yes_probs": yes_logprobs,
        "No_probs": no_logprobs,
        "A_probs": a_logprobs,
        "B_probs": b_logprobs,
    }

    del model
    model_cleanup()
    return results_dict

# class OutputType(Enum):
#     TEXT = "text"
#     LOGPROBS = "logprobs"

def hf_auth_check(model_list):
    for model in model_list:
        auth_check(model)


def _test(seed):
    prompts_dataset = Dataset.from_dict(get_dataset_for_coverage_questions()[:2])
    for model_name in ["meta-llama/Llama-3.1-8B-Instruct"]:
        results_dict = load_and_infer_with_model(model_name, seed, prompts_dataset)
        results = Dataset.from_dict(results_dict)
        print(f"For {model_name} results :{results}")
        results.to_csv(f"tests/{model_name}-results.csv", index=False)

def main(seed):
    prompts_dataset = get_dataset_for_coverage_questions()
    print("Using the following dataset:")
    print(prompts_dataset)
    for model_name in tqdm(model_list, desc="For models"):
        print("Running with model:", model_name)
        results_dict = load_and_infer_with_model(model_name, seed, prompts_dataset)
        # TODO see if we should just make it pyarrow
        results = Dataset.from_dict(results_dict)
        print(f"For {model_name} results :{results}")
        results.to_csv(f"runs/{model_name}-results.csv", index=False)
        del results
        del results_dict
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", default=False, action="store_true")
    args = parser.parse_args()
    if args.test:
        _test(args.seed)
        exit(1)

    main(args.seed)
