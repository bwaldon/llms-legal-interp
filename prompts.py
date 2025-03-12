import pandas as pd
from datasets import load_dataset
SEED = 42
contracts_file = "data/vague_contracts.csv"
dataset = load_dataset("csv", data_files={
        "test": "data/vague_contracts.csv",
})
dev_dataset = dataset.shuffle(seed=SEED)["test"][:5]


def yes_or_no_prompt(item):
    return f"""{item.header}
    {item.continuation}
    {item.version}
    Do you think that the claim is covered under {item.locus_of_uncertainty} as it appears in the policy?
    """
def make_it_do_you_agree(item, actor = "Insurance Company"):
    return f"""{item.header}
    {item.continuation}
    {actor} decided that the claim is {item.version} under the policy.
    Do you agree?
    """
