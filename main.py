from vllm import LLM, SamplingParams
import csv
import gc
import torch

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# llm = LLM(model="facebook/opt-125m")
model_list = [
    # "meta-llama/Llama-3.2-1B",
    # "meta-llama/Llama-3.2-3B",
    # "meta-llama/Llama-3.2-8B",
    # "gpt2-medium",
    # "gpt2-large",
    # "gpt2-xl",
    "bigscience/bloom-560m"
    "bigscience/bloom-1b1",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1"
]


def get_relevant_fields(row):
    return row[3], row[4], row[11]

with open("vague_contracts/data/1_falseconsensus/demo-merged.csv") as f:
    r = csv.reader(f, delimiter=",", quotechar='"')
    data = [row for row in r]

context, policy, ambiguity = get_relevant_fields(data[1])

questions = [
    "Should the insurer have accepted the claim? Please answer with either yes or no.",
    "Shouldn't the insurer have accepted the claim? Please answer with either yes or no.",
    "Should the insurer have accepted the claim? Please answer with either no or yes.",
    "Shouldn't the insurer have accepted the claim? Please answer with either no or yes.",
    "Should the insurer not accept the claim? Please answer with either no or yes.",
    "Should the insurer have denied the claim? Please answer with either yes or no.",
    "Shouldn't the insurer have denied the claim? Please answer with either yes or no.",
    "Should the insurer have denied the claim? Please answer with either no or yes.",
    "Shouldn't the insurer have denied the claim? Please answer with either no or yes.",
    "Should the insurer not deny the claim? Please answer with either no or yes.",
    "Wasn't the insurer correct in their decision to accept the claim?"
]

for model in model_list:
    print("**************************")
    print(model)

    llm = LLM(model=model)

    prompts = [
        (f"Consider the context: {context}"
         f""
         f"Consider the insurance policy: {policy}"
         f""
         f"{question}")
        for question in questions
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(generated_text)

    del llm
    torch.cuda.empty_cache()
    gc.collect()