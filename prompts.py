from typing import Literal
import re

from datasets import Dataset, load_dataset


def get_prompt_dataset(prompt_function, prompt_type):
    def convert_header_to_definition(header):
        print(header)
        match re.findall(r"<b>[\"\s]*(.*?)[\"\s]*</?b>", header):
            case [x]:
                return x
            case [x, y]:
                return x + y
            case _:
                return None

    VAGUE_CONTRACTS_FILE = "data/vague_contracts.csv"
    vague_contracts_dataset = load_dataset("csv", data_files={"test": VAGUE_CONTRACTS_FILE})
    vague_contracts_dataset["test"] = vague_contracts_dataset["test"].map(
        lambda x: {"definition": convert_header_to_definition(x["header"])}, batched=False)
    print(vague_contracts_dataset.shape)
    vague_contracts_prompts = vague_contracts_dataset.map(lambda x: {
        "prompt": prompt_function(x),
        "prompt_type": prompt_type,
    })
    return vague_contracts_prompts

#1 Binary (Meaning) Judgements
# Question style similar to summarization from Blair-Stanek et al.
def binary_option_question(scenario, definition, options):
    return f"""Consider the following:
        {scenario} 
        Does it contain an instance of {definition}? Answer with ONLY the string {options[0]} or {options[1]}:   
        """

def no_or_yes_question(item):
    return binary_option_question(item["continuation"], item["definition"], ['No', 'Yes'])


def yes_or_no_question(item):
    return binary_option_question(item["continuation"], item["definition"], ['Yes', 'No'])


def make_it_do_you_agree(item, actor="Insurance Company"):
    return f"""{item.header}
    {item.continuation}
    {actor} decided that the claim is {item.version} under the policy.
    Do you agree?
    """

def get_yes_or_no_vague_contracts():
    return get_prompt_dataset(yes_or_no_question, "yes_or_no")

if __name__ == "__main__":
    dataset = get_yes_or_no_vague_contracts()["test"]
    dataset.to_csv("prompts/yes_or_no_vague_contracts.csv")
