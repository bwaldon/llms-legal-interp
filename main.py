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
    # "gpt2-large" fails at 2 prompts,  logbrobs 16
    # "gpt2-xl", fails at 2 prompts,  logbrobs 16
    #"bigscience/bloom-560m", Works
    #"bigscience/bloom-1b1", fails at 2 prompts, logbrobs 16
    #"bigscience/bloom-3b",  fails at 2 prompts, logbrobs 16
    #"bigscience/bloom-7b1", fails at 2 prompts, logbrobs 16
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
    Interactive snippet for inference
    prompt = MetaLinguisticPrompt(topic="landscaping", features=["question", "bool"])
    model = MetaLinguisticJudgement("meta-llama/Llama-3.2-1B", 42)
    model.infer([prompt])
    """


def load_and_infer_with_model(model_name, seed, dataset, tokens_of_interest=("Yes", "No", "A", "B")):
    def model_cleanup():
        torch.cuda.empty_cache()
        gc.collect()


    print(model_name + "\n")

    model = MetaLinguisticJudgement(model_name, seed)
    def get_token_ids_of_interest(tokens_of_interest, vocab):
        token_ids = dict()
        for token in tokens_of_interest:
            token_ids[token] =  None #list()
            if token in vocab:
                token_ids[token] = (vocab[token])

            # if token.lower() in vocab:
            #     token_ids[token].append(vocab[token.lower()])
        return token_ids

    token_ids_of_interest = get_token_ids_of_interest(tokens_of_interest, model.tokenizer.get_vocab())
    prompts = dataset["prompt"]
    outputs = model.infer(prompts)

    # Collate data and outputs
    def collate_data_and_outputs(model_name, dataset, outputs):

        # https://stackoverflow.com/questions/43647186/tokenize-based-on-white-space-and-trailing-punctuation
        def extract_first_answer_token(text):
            return [x.strip("\"'\.!") for x in re.split(r"([a-zA-z]+)?\s+", text) if x][0]

        def collate_logprobs_for_tokens_of_interest(n, token_ids_of_interest, output):
            seq_len = len(output.logprobs)
            tokens_probs = list()
            for token in token_ids_of_interest:
                tokens_probs[token] = np.ndarray((seq_len, 1))

            for position, logprobs in enumerate(output.logprobs)[:n]:
                for token, token_id in token_ids_of_interest.items():
                    tokens_probs[token][position] = logprobs[token_id].logprob if token_id in logprobs else 0
            return tokens_probs

        print(f"Dataset : {len(dataset)} {len(outputs)}")
        # Just get the probs for the output
        # TODO batched processing
        results_token_probs = [collate_logprobs_for_tokens_of_interest(1, token_ids_of_interest, output) for output in outputs]
        results_dict = {
            "title": dataset["title"],
            "prompt_type": dataset["prompt_type"],
            "prompt": dataset["prompt"],
            "version": dataset["version"],
            "output": [extract_first_answer_token(output.text) for output in outputs],
            "output_text": [output.text for output in outputs],
            "cum_logprob": [output.cumulative_logprob for output in outputs]
        }

        results_dict["Yes_prob"] = [token_probs["Yes"] for token_probs in results_token_probs]
        results_dict["No_prob"] = [token_probs["No"] for token_probs in results_token_probs]
        results_dict["A_prob"] = [token_probs["A"] for token_probs in results_token_probs]
        results_dict["B_prob"] = [token_probs["B"] for token_probs in results_token_probs]
        return results_dict

    del model
    model_cleanup()
    return collate_data_and_outputs(model_name, dataset, outputs)

# class OutputType(Enum):
#     TEXT = "text"
#     LOGPROBS = "logprobs"

def hf_auth_check(model_list):
    for model in model_list:
        auth_check(model)


def test(seed):
    prompts_dataset = Dataset.from_dict(get_dataset_for_coverage_questions()[:2])
    for model_name in model_list[:1]:
        load_and_infer_with_model(model_name, seed, prompts_dataset)

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
        test(args.seed)
        exit(1)

    main(args.seed)
