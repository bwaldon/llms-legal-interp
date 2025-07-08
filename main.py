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
from model import MetaLinguisticJudgement
from datasets import Dataset
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    # Main set of models with instruct divide and size variety
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    # Small reference model - would allow for pretraining variation
    "gpt2-medium",
    # Other open models
    "allenai/OLMo-2-1124-7B-Instruct",
    "mistralai/Ministral-8B-Instruct-2410",
    "google/gemma-7b-it"
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

    prompts = dataset["prompt"]
    # TODO combine probability extraction and relevant tokens into a single flow
    tokenizer = model.llm.get_tokenizer()
    outputs = model.infer(prompts)
    model_cleanup()
    yes_logprobs, no_logprobs, a_logprobs, b_logprobs = model.probs(prompts)

    results_dict = {
        "title": dataset["title"],
        "prompt_type": dataset["prompt_type"],
        "prompt": dataset["prompt"],
        "version": dataset["version"],
        "output": tokenizer.batch_decode([output.token_ids[0] for output in outputs]),
        "output_text": [output.text for output in outputs],
        "cum_logprob": [output.cumulative_logprob for output in outputs],
        "Yes_probs": yes_logprobs,
        "No_probs": no_logprobs,
        "A_probs": a_logprobs,
        "B_probs": b_logprobs,
    }

    del model
    model_cleanup()
    return results_dict

def hf_auth_check(model_list):
    for model in model_list:
        auth_check(model)



def _test(seed):
    prompts_dataset = Dataset.from_dict(get_dataset_for_coverage_questions()[:9])
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
        # TODO parameterize with seed
        print(f"For {model_name} results :{results}")

        results.to_csv(f"runs-{seed}/{model_name}-results.csv", index=False)
        del results
        del results_dict
        torch.cuda.empty_cache()
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
