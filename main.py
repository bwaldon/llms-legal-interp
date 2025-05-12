import argparse

from model import MetaLinguisticPrompt, MetaLinguisticJudgement
from huggingface_hub import login
import csv
import gc
import torch
import random
import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def main(seed):

    model_list = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        # "meta-llama/Llama-3.2-8B",
        # "gpt2-medium",
        # "gpt2-large",
        # "gpt2-xl",
        # "bigscience/bloom-560m"
        # "bigscience/bloom-1b1",
        # "bigscience/bloom-3b",
        # "bigscience/bloom-7b1"
    ]

    login(token=open("hf_token.txt").read())

    for model_name in model_list:

        model = MetaLinguisticJudgement(model_name, seed)
        prompts = [
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

        outputs = model.infer(prompts)

        print(model_name + "\n")

        for p, o in zip(prompts, outputs):
            print(f"*********************{p.topic}********************{p.features}*************************\n")
            print(p.text)
            print("\n\n")
            print(o)
            print("\n\n\n")


        del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)