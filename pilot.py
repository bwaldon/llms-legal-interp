"""

OpenAI supports both single and batch requests.
Ref: https://github.com/Aatlantise/advarsarial-nli-amr/blob/main/open_ai_api.py
Ref: https://platform.openai.com/docs/guides/batch
Ref: https://platform.openai.com/docs/api-reference/completions

"""

import os

from functools import reduce
from collections import OrderedDict

import argparse
from pathlib import Path
import json
import numpy as np
from datasets import Dataset
from openai import OpenAI

from prompts import get_dataset_for_coverage_questions


# def infer_from_openai(prompts, model="gpt-4", max_tokens=100, temperature=0.0):
#     client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#     # Completions using the 'legacy' completions endpoint

def request_batch(prompts, batch_name, model="gpt-4-0613", max_tokens=1, temperature=0.0):
    def completion_request_for_prompt(prompt, prompt_id):
        return { "custom_id": prompt_id, "method": "POST", "url": "/v1/chat/completions",
                    "body": {
                        "model": model, # gpt-4o-2024-11-20
                        "prompt": prompt,
                        "max_tokens": 1,
                        "temperature": 0,
                    }
        }

    requests_batch = [completion_request_for_prompt(prompt, batch_name + "-" + str(idx)) for idx, prompt in enumerate(prompts)]
    return requests_batch

def infer_and_extract_output(client, prompt_list):
    def response(prompt):
        return client.responses.create(
            model="gpt-4-0613",
            #messages=[{"role": "user", "content": prompt}],
            input=prompt,
            # No use of prompt template right now
            include=["message.output_text.logprobs"],
            max_output_tokens=16,
            temperature=0,
            top_logprobs=20,
        )

    responses = [response(prompt) for prompt in prompt_list]
    def output_from_response(response):
        content = response.output[0].content[0],
        print(type(content[0]))
        print(content[0])
        return {
            "text": content[0].text,
            "logprobs" : OrderedDict((lp.token, lp.logprob) for lp in content[0].logprobs[0].top_logprobs)
        }
    outputs = [output_from_response(response) for response in responses]
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test", default=False, action="store_true")

    args = parser.parse_args()
    if args.test:
        prompts_dataset = get_dataset_for_coverage_questions()[:9]
    else:
        prompts_dataset = get_dataset_for_coverage_questions()

    prompts_list = prompts_dataset['prompt']

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    outputs = infer_and_extract_output(client, prompts_list)

    if len(outputs) != len(prompts_dataset):
        print(f"!!!! Length mismatch: {len(outputs)} outputs vs {len(prompts_dataset)} dataset")


    def collate_outputs(dataset, outputs):
        # TODO abhip: unify candidate variables with model.py
        # Candidates copied from model.py
        candidates = ['YES', 'Yes', 'yes', 'NO', 'No', 'no', 'A', 'B']
        MIN_FLOAT = np.finfo(float).min


        results_dict = {
            "title": dataset["title"],
            "prompt_type": dataset["prompt_type"],
            "prompt": dataset["prompt"],
            "version": dataset["version"],
            "output": [output["logprobs"].popitem(last=False)[0] for output in outputs],
            "output_text": [output["text"] for output in outputs],
            # "cum_logprob": [output.cumulative_logprob for output in outputs],
        }

        candidate_logprobs = OrderedDict()

        for candidate in candidates:
            candidate_logprobs[candidate] = list()

        for output in outputs:
            for candidate in candidates:
                candidate_logprobs[candidate].append(
                    output["logprobs"].get(candidate, MIN_FLOAT)
                )

        for candidate in candidate_logprobs:
            results_dict[candidate + "_probs"] = candidate_logprobs[candidate]

        return results_dict


    results_dict = collate_outputs(prompts_dataset, outputs)
    results = Dataset.from_dict(results_dict)
    # Save to JSON
    results.to_json(args.out.with_suffix(".json"), index=False)
    # Save to CSV
    results.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"Saved results to {args.out.with_suffix('.json')} and {args.out.with_suffix('.csv')}")
