import csv
import re
from datasets import load_dataset, concatenate_datasets
from enum import Enum
from typing import Literal
from collections import OrderedDict
from pprint import pprint
from tqdm import tqdm


class MetaLinguisticPrompt:
    def __init__(self, nudge=None, features=None, topic="landscaping"):
        self.nudge = nudge
        self.features = features
        self.topic = topic

        self.get_base_text()

        if features is not None:
            self.add_feature()

        if nudge is not None:
            self.add_nudge()

    def get_base_text(self):
        if self.topic == "landscaping":
            if "question" in self.features:
                self.text = ('Can you consider the ordinary meaning of "landscaping",'
                         ' and decide whether installing an in-ground trampoline would be included in such meaning?')
            elif "arg" in self.features:
                self.text = ('Considering the ordinary meaning of "landscaping", '
                             'installing an in-ground trampoline is landscaping.')
            elif "neg_arg" in self.features:
                self.text = ('Considering the ordinary meaning of "landscaping", '
                             'installing an in-ground trampoline is not landscaping.')
            elif "event" in self.features:
                self.text = ('Consider the ordinary meaning of "landscaping". '
                             'Abhishek was installing an in-ground trampoline in his backyard, when he suffered an injury.')
        elif self.topic == "restraining":
            if "question" in self.features:
                self.text = ('Can you tell me the ordinary meaning of "physically restraining someone",'
                             ' and whether holding someone at gunpoint and telling them not to move would be included in such meaning?')
            elif "arg" in self.features:
                self.text = ('Considering the ordinary meaning of "physically restraining someone", '
                             'holding someone at gunpoint and telling them not to move is landscaping.')
            elif "neg_arg" in self.features:
                self.text = ('Considering the ordinary meaning of "physically restraining someone", '
                             'holding someone at gunpoint and telling them not to move is not landscaping.')
            elif "event" in self.features:
                self.text = ('Consider the ordinary meaning of "physically restraining someone". '
                             'Hyun was being held at gunpoint and told not to move.')
        elif self.topic == "vague":
            self.text: str = self.from_vague_contracts()

    def add_feature(self):
        if "coverage" in self.features:
            self.text += " His insurance covers losses incurred from landscaping-related events, including injuries during installation."

        if "judgement" in self.features:
            self.text += " Therefore, his injury is covered."
        elif "neg_judgement" in self.features:
            self.text += " Therefore, his injury is not covered."

        if "bool" in self.features:
            if "reverse" in self.features:
                self.text += " No, or yes?"
            else:
                self.text += " Yes, or no?"
        elif "q_agree" in self.features:
            if "neg" in self.features:
                self.text += " Do you disagree?"
            else:
                self.text += " Do you agree?"
        elif "q_cover" in self.features:
            self.text += " Is his injury covered under his insurance?"
        elif "neg_q_cover" in self.features:
            self.text += " Is his injury not covered under his insurance?"
        elif "mc_prompt" in self.features:
            self.text += " Which is more likely? 1. His injury is covered. 2. His injury is not covered."
        elif "mc_prompt_reverse" in self.features:
            self.text += " Which is more likely? 1. His injury is not covered. 2. His injury is covered."

    def add_nudge(self):
        if self.nudge is None:
            pass
        elif self.nudge == "Yes":
            self.text += " If yes, what would make it so?"
        elif self.nudge == "No":
            self.text += " If no, why not?"
        else:
            pass

    def from_vague_contracts(self):
        #todo: implement vague_contracts pipeline
        with open("data/clean/vague_contracts.csv") as f:
            r = csv.reader(f, delimiter=",", quotechar='"')
            data = [row for row in r]

        context, policy, ambiguity = data[3], data[4], data[11]

        return ""

def locus_premise(locus_of_uncertainty):
    return f"Considering just how the word \"{locus_of_uncertainty}\" would be understood by ordinary speakers of English"

## Coverage Prompt functions - posing the questions with the proxy of coverage
def is_person_covered_question(person_name):
    return f"Is {person_name} covered by the insurance?"


def coverage_binary_question(binary_question_suffix, contract):
    # list for HF
    return f"""{contract['header']}
{contract['continuation']}
{locus_premise(contract['locus_of_uncertainty'])}, {is_person_covered_question(contract['person_name'])} {binary_question_suffix}"""


ANSWER_TRIGGER="Final answer is:"
YES_NO_QUESTION = f"Yes or No? {ANSWER_TRIGGER}"
NO_YES_QUESTION = f"No or Yes? {ANSWER_TRIGGER}"


def coverage_binary_question_yes_no(contract):
    return coverage_binary_question(YES_NO_QUESTION, contract)


def coverage_binary_question_no_yes(contract):
    return coverage_binary_question(NO_YES_QUESTION, contract)

def is_person_not_covered_question(person_name):
    return f"Is {person_name} not covered by the insurance?"

def coverage_binary_question_negation(contract):
    # list for HF
    return f"""{contract['header']}
{contract['continuation']}
{locus_premise(contract['locus_of_uncertainty'])}, {is_person_not_covered_question(contract['person_name'])}? {YES_NO_QUESTION}"""

def coverage_agreement(contract):
    return f"""{contract['header']}
    {contract['continuation']}
    {locus_premise(contract['locus_of_uncertainty'])}, {contract['person_name']} is covered by the insurance. Do you agree? {YES_NO_QUESTION}"""


def coverage_agreement_on_negation(contract):
    return f"""{contract['header']}
    {contract['continuation']}
    {locus_premise(contract['locus_of_uncertainty'])}, {contract['person_name']} is not covered by the insurance. Do you agree? {YES_NO_QUESTION}"""


def coverage_disagreement(contract):
    return f"""{contract['header']}
    {contract['continuation']}
    {locus_premise(contract['locus_of_uncertainty'])}, {contract['person_name']} is covered by the insurance. Do you disagree? {YES_NO_QUESTION}"""


def coverage_disagreement_on_negation(contract):
    return f"""{contract['header']}
    {contract['continuation']}
    {locus_premise(contract['locus_of_uncertainty'])}, {contract['person_name']} is not covered by the insurance. Do you disagree? {YES_NO_QUESTION}"""


def coverage_options(contract):
    return f"""{contract['header']}
        {contract['continuation']}
    {locus_premise(contract['locus_of_uncertainty'])}, {is_person_covered_question(contract['person_name'])} Options: A. {contract['person_name']} is covered. B. {contract['person_name']} is not covered. {ANSWER_TRIGGER}"""

def coverage_options_flipped(contract):
    return f"""{contract['header']}
        {contract['continuation']}
    {locus_premise(contract['locus_of_uncertainty'])}, {is_person_covered_question(contract['person_name'])} Options: A. {contract['person_name']} is not covered. B. {contract['person_name']} is covered. {ANSWER_TRIGGER}"""


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
    dataset.to_csv("data/prompts/coverage_8_vague_contracts_prompts.csv")
