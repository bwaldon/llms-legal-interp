import gc
import csv

import torch
from huggingface_hub import auth_check

from model import MetaLinguisticPrompt, JudgementModel
from prompts import get_yes_or_no_vague_contracts


model_list = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B",
    # "meta-llama/Llama-3.2-3B",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "meta-llama/Llama-3.2-8B",
    # "gpt2-medium",
    # "gpt2-large",
    # "gpt2-xl",
    # "bigscience/bloom-560m"
    # "bigscience/bloom-1b1",
    # "bigscience/bloom-3b",
    # "bigscience/bloom-7b1"
]


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def get_prompts():
    return [
        MetaLinguisticPrompt(topic="landscaping", nudge="No"),
        MetaLinguisticPrompt(topic="landscaping", nudge=None),
        MetaLinguisticPrompt(topic="landscaping", nudge="Yes"),
        MetaLinguisticPrompt(topic="restraining", nudge="No"),
        MetaLinguisticPrompt(topic="restraining", nudge=None),
        MetaLinguisticPrompt(topic="restraining", nudge="Yes"),
    ]


def load_and_infer_with_model(model_name, prompts, dataset):
    print(model_name + "\n")
    model = JudgementModel(model_name)
    outputs = model.infer(dataset["prompt"])

    def print_prompts_and_output(p, o):
        star_20 = '*' * 20
        print(f"{star_20}{p.topic}{star_20}{p.nudge}{star_20}", end="\n\n")
        print(p.text)
        print("\n\n")
        print(o)
        print("\n\n\n")

    for p, o in zip(prompts, outputs):
        print_prompts_and_output(p, o)
    del model
    cleanup()

def hf_auth_check(model_list):
    for model in model_list:
        auth_check(model)

if __name__ == "__main__":
    hf_auth_check(model_list)
    # TODO make it a dataset
    prompts = get_prompts()
    prompts_dataset = get_yes_or_no_vague_contracts()["test"]
    for model_name in model_list:
        load_and_infer_with_model(model_name, prompts, prompts_dataset)
