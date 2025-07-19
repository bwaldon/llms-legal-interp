import csv
import re
from datasets import load_dataset, concatenate_datasets
from enum import Enum
from typing import Literal
from collections import OrderedDict
from pprint import pprint
from tqdm import tqdm

def locus_premise(locus_of_uncertainty):
    locus_of_uncertainty = locus_of_uncertainty.strip()
    if locus_of_uncertainty == 'causative "from"':
        return f"Considering just how {locus_of_uncertainty} would be understood by ordinary speakers of English"
    return f"Considering just how \"{locus_of_uncertainty}\" would be understood by ordinary speakers of English"

## Coverage Prompt functions - posing the questions with the proxy of coverage
def is_person_covered_question_add_on(person_name):
    return f"is {person_name} covered by the insurance"


def coverage_binary_question(binary_question_suffix, contract):
    # list for HF
    return f"""{contract['header']}
{contract['continuation']}
{locus_premise(contract['locus_of_uncertainty'])}, {is_person_covered_question_add_on(contract['person_name'])}\u2014{binary_question_suffix}"""


ANSWER_TRIGGER="Final answer is:"
YES_NO_QUESTION = f"yes or no? {ANSWER_TRIGGER}"
NO_YES_QUESTION = f"no or yes? {ANSWER_TRIGGER}"


def coverage_binary_question_yes_no(contract):
    return coverage_binary_question(YES_NO_QUESTION, contract)


def coverage_binary_question_no_yes(contract):
    return coverage_binary_question(NO_YES_QUESTION, contract)

def is_person_not_covered_question(person_name):
    return f"is {person_name} not covered by the insurance"

def coverage_binary_question_negation(contract):
    # list for HF
    return f"""{contract['header']}
{contract['continuation']}
{locus_premise(contract['locus_of_uncertainty'])}, {is_person_not_covered_question(contract['person_name'])}\u2014{YES_NO_QUESTION}"""
def agree_disagree_question_add_on(agreement: Literal["agree", "disagree"]):
    return f"do you {agreement} with the following statement"

def coverage_agreement(contract):
    return f"""{contract['header']}
{contract['continuation']}
{locus_premise(contract['locus_of_uncertainty'])}, {agree_disagree_question_add_on('agree')}: "{contract['person_name']} is covered by the insurance"\u2014{YES_NO_QUESTION}"""


def coverage_agreement_on_negation(contract):
    return f"""{contract['header']}
{contract['continuation']}
{locus_premise(contract['locus_of_uncertainty'])}, {agree_disagree_question_add_on('agree')}: "{contract['person_name']} is not covered by the insurance"\u2014{YES_NO_QUESTION}"""


def coverage_disagreement(contract):
    return f"""{contract['header']}
{contract['continuation']}
{locus_premise(contract['locus_of_uncertainty'])}, {agree_disagree_question_add_on('disagree')}: "{contract['person_name']} is covered by the insurance"\u2014{YES_NO_QUESTION}"""


def coverage_disagreement_on_negation(contract):
    return f"""{contract['header']}
{contract['continuation']}
{locus_premise(contract['locus_of_uncertainty'])}, {agree_disagree_question_add_on('disagree')}: "{contract['person_name']} is not covered by the insurance"\u2014{YES_NO_QUESTION}"""


def coverage_options(contract):
    return f"""{contract['header']}
{contract['continuation']}
{locus_premise(contract['locus_of_uncertainty'])}, {is_person_covered_question_add_on(contract['person_name'])}? Options: A. {contract['person_name']} is covered. B. {contract['person_name']} is not covered. {ANSWER_TRIGGER}"""

def coverage_options_flipped(contract):
    return f"""{contract['header']}
{contract['continuation']}
{locus_premise(contract['locus_of_uncertainty'])}, {is_person_covered_question_add_on(contract['person_name'])}? Options: A. {contract['person_name']} is not covered. B. {contract['person_name']} is covered. {ANSWER_TRIGGER}"""


def construct_dataset(prompt_types):
    VAGUE_CONTRACTS_FILE = "data/clean/vague_contracts.csv"
    vague_contracts_dataset = load_dataset("csv", data_files={"test": VAGUE_CONTRACTS_FILE})
    prompt_datasets = list()
    for prompt_type, prompt_function in tqdm(prompt_types.items()):
        prompts_dataset = vague_contracts_dataset["test"].map(lambda x: {
            "prompt": prompt_function(x),
            "prompt_type": prompt_type
        })
        print(prompt_type, prompts_dataset)
        prompt_datasets.append(prompts_dataset)
    return concatenate_datasets(prompt_datasets)


prompt_types = ["yes_or_no", "no_or_yes", "negation", "agreement", "agreement_negation", "disagreement", "disagreement_negation", "options", "options_flipped"]
prompt_template_functions = {
    "yes_or_no": coverage_binary_question_yes_no,
    "no_or_yes": coverage_binary_question_no_yes,
    "negation": coverage_binary_question_negation,
    "agreement": coverage_agreement,
    "agreement_negation": coverage_agreement_on_negation,
    "disagreement": coverage_disagreement,
    "disagreement_negation": coverage_disagreement_on_negation,
    "options": coverage_options,
    "options_flipped": coverage_options_flipped,
}

def get_dataset_for_coverage_questions():
    return construct_dataset(
       prompt_template_functions
    )

if __name__ == "__main__":
    dataset = get_dataset_for_coverage_questions()
    print(dataset)
    dataset.to_csv("data/prompts/coverage_contracts.csv")
