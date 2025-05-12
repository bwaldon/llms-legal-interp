import csv
from typing import Literal
import re
from pprint import pprint

from datasets import Dataset, IterableDataset, load_dataset

def get_prompt_dataset(prompt_function, prompt_type):
    VAGUE_CONTRACTS_FILE = "data/clean/vague_contracts.csv"
    vague_contracts_dataset = load_dataset("csv", data_files={"test": VAGUE_CONTRACTS_FILE})
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


# def make_it_do_you_agree(item, actor="Insurance Company"):
#     return f"""{item.header}
#     {item.continuation}
#     {actor} decided that the claim is {item.version} under the policy.
#     Do you agree?
#     """

def get_yes_or_no_vague_contracts():
    return get_prompt_dataset(yes_or_no_question, "yes_or_no")

## Coverage Prompt functions - posing the questions with the proxy of coverage

def coverage_binary_question(contract):
    YES_OR_NO_PROMPT = "Answer with ONLY the string Yes or the string No"
    return {"prompt": f"""{contract['header']}
    {contract['continuation']}
    Considering just how the word "{contract['locus_of_uncertainty']}" would be understood by ordinary speakers of English, Is {contract['person_name']} covered by the John covered by the insurance? {YES_OR_NO_PROMPT}
    """}

class MetaLinguisticPrompt:
    def __init__(self, nudge=None, features=None, topic="landscaping"):
        self.nudge = nudge
        self.features = features
        self.topic = topic

        self.text = self.__make_prompt()
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
if __name__ == "__main__":
    # dataset = get_yes_or_no_vague_contracts()
    # dataset.to_csv("data/prompts/yes_or_no_vague_contracts_prompts.csv")
    landscaping_dataset = Dataset.from_csv("data/landscaping.csv")
    dataset = landscaping_dataset.map(coverage_binary_question)
    for item in dataset:
        pprint(item)






