from model import MetaLinguisticPrompt, MetaLinguisticJudgement
from huggingface_hub import login
import csv
import gc
import torch

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

login(token=open("hf_token.txt").read())

for model_name in model_list:

    model = MetaLinguisticJudgement(model_name)
    prompts = [
        MetaLinguisticPrompt(topic="landscaping", nudge="No"),
        MetaLinguisticPrompt(topic="landscaping", nudge=None),
        MetaLinguisticPrompt(topic="landscaping", nudge="Yes"),
        MetaLinguisticPrompt(topic="restraining", nudge="No"),
        MetaLinguisticPrompt(topic="restraining", nudge=None),
        MetaLinguisticPrompt(topic="restraining", nudge="Yes"),
    ]

    outputs = model.infer(prompts)

    print(model_name + "\n")

    for p, o in zip(prompts, outputs):
        print(f"*********************{p.topic}********************{p.nudge}*************************\n")
        print(p.text)
        print("\n\n")
        print(o)
        print("\n\n\n")


    del model
    torch.cuda.empty_cache()
    gc.collect()