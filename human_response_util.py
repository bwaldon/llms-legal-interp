import pandas as pd
import numpy as np


if __name__ ==  "__main__":
    human_responses = pd.read_csv("data/human/main-merged.csv")
    # items in which item majority is COVERED
    # print(human_responses.groupby(['title', 'version', 'item'], as_index=False)
    #       .aggregate({'individual_judgment': 'count'})
    # )
    print(human_responses[['title', 'version', 'item', 'individual_judgment']].head())
    human_counts = human_responses.groupby(['title', 'version', 'item'], as_index=False).aggregate({'individual_judgment':'count'})

    print("human_counts", human_counts.shape)

    # Ignore "cantdecide" responses
    human_responses = human_responses[human_responses["individual_judgment"] != "cantdecide"]

    yes_index = human_responses["individual_judgment"] == "yes"
    no_index = human_responses["individual_judgment"] == "no"


    yes_responses = human_responses[yes_index]
    yes_counts = yes_responses.groupby(['title', 'version', 'item'], as_index=False).aggregate({'individual_judgment':'count'})
    print("yes_counts", yes_counts.shape)

    no_responses = human_responses[no_index]
    no_counts = no_responses.groupby(['title', 'version', 'item'], as_index=False).aggregate({'individual_judgment':'count'})
    print("no_counts", no_counts.shape)

    covered_proportions = yes_counts["individual_judgment"] / human_counts["individual_judgment"]
    not_covered_proportions = no_counts["individual_judgment"] / human_counts["individual_judgment"]

    covered_majority = covered_proportions > 0.5
    print(np.sum(covered_majority), "items have majority COVERED")
    not_covered_majority = covered_proportions <= 0.5

    not_covered_responses_majority = not_covered_proportions >= 0.5
    print(np.sum(not_covered_majority), "items have majority NOT COVERED")
    print(np.sum(not_covered_responses_majority), "items have majority NOT COVERED explicitly")

    merged_counts = pd.merge(human_counts, yes_counts, on=['title', 'version', 'item'], how='left', suffixes=('_total', '_yes'))
    merged_counts = merged_counts.fillna(0)
    print(merged_counts.shape)

    merged_counts["covered_proportion"] = merged_counts["individual_judgment_yes"] / merged_counts["individual_judgment_total"]
    merged_counts["covered_majority"] = merged_counts["covered_proportion"] > 0.5
    print(np.sum(merged_counts["covered_majority"]), "items have majority COVERED (from merged)")
