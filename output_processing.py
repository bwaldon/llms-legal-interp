import pandas as pd

from prompts import prompt_types, candidates
import numpy as np
import scipy as sp

def get_aff_unaff_columns(prompt_type):
    match prompt_type:
        case "yes_or_no" | "no_or_yes" | "agreement" | "disagreement_negation" | "negation":
            return "Yes_probs", "No_probs"
        case "disagreement" | "agreement_negation" :
            return "No_probs", "Yes_probs"
        case "options":
            return "A_probs", "B_probs"
        case "options_flipped":
            return "B_probs", "A_probs"
        case _:
            raise ValueError()

def conditional_entropy(PX, PY):
    # Conditional Entropy
    # https://datascience.stackexchange.com/questions/58565/conditional-entropy-calculation-in-python-hyx
    # https://www.cs.cmu.edu/~venkatg/teaching/ITCS-spr2013/notes/lect-jan17.pdf
    # all pairs x in X, y in
    joint_probs = np.outer(PX, PY)
    conditional_probs = np.divide(joint_probs, PX, where=PX!=0, out=np.zeros_like(joint_probs))
    # log_term = np.log(conditional_probs, where=conditional_probs!= 0, out=np.zeros_like(conditional_probs))
    return -np.sum(np.where(joint_probs !=0, joint_probs * log_term, 0)) # scalar

def exponentiate_fields(df, fields):
    for field in fields:
        df[field] = df[field].transform(np.exp)
    return df

def organize_distribution(model_results):
    # Candidate probs fields are logprobs, so we exponentiate them
    # NOTE: this is a copy of the candidates list in model.py - check for match
    model_results = exponentiate_fields(model_results, [x + "_probs" for x in candidates])
    # Summate yes and no candidate probs
    model_results["Yes_probs"] = model_results["Yes_probs"] + model_results["yes_probs"] + model_results["YES_probs"]
    model_results["No_probs"] = model_results["No_probs"] + model_results["no_probs"] + model_results["NO_probs"]

    model_results["Other_prob"] = 1 - model_results["Yes_probs"] - model_results["No_probs"]

    for group, indices in model_results.groupby("prompt_type").indices.items():
        aff_column, unaff_column = get_aff_unaff_columns(group)
        model_results.loc[indices, "Aff_prob"] = model_results[aff_column][indices]
        model_results.loc[indices, "UnAff_prob"] = model_results[unaff_column][indices]
        def normalize(x, y):
            return x/(x+y), y/(x+y)
q        model_results["Covered_prob"], model_results["NotCovered_prob"] = model_results["Aff_prob"], model_results["UnAff_prob"]
        model_results["Covered"] = model_results["Aff_prob"] > model_results["UnAff_prob"]
        model_results["NotCovered"] = model_results["Aff_prob"] <= model_results["UnAff_prob"]

    model_results["entropy"] = model_results[["Aff_prob", "UnAff_prob", "Other_prob"]].apply(lambda x: sp.stats.entropy(x), axis=1)

    return model_results

def calculate_relative_measures(model_results):
    relative_measures = {
        "title": list(),
        "version": list(),
        "model_name": list(),
        "prompt_type": list(),
        "js_dist": list(),
        "kl_div": list(),
        # "more_than_reversal": list(),
        # "cond_entropy": list()
    }
    # control and variation pairs
    for group, df in model_results.groupby(['title', 'version', 'model_name'], sort=False):
        control_mask = df.prompt_type == "yes_or_no"
        control_probs = df[["Covered_prob", "NotCovered_prob", "Other_prob"]][control_mask]

        # reversal_probs = df[["UnAff_prob", "Aff_prob", "Other_prob"]][control_mask]
        # reversal_dists = np.array([sp.spatial.distance.jensenshannon(x, y) for x, y in zip(control_probs.values, reversal_probs.values)])
        for variation in prompt_types[1:]:
            relative_measures["title"].append(group[0])
            relative_measures["version"].append(group[1])
            relative_measures["model_name"].append(group[2])
            relative_measures["prompt_type"].append(variation)

            variation_mask = df.prompt_type == variation
            variation_probs = df[["Covered_prob", "NotCovered_prob", "Other_prob"]][variation_mask]
            # # TODO numpy it
            # relative_measures["cond_entropy"].append(conditional_entropy(control_probs.values, variation_probs.values))

            # Jensen-Shannon Distance
            js_dist = sp.spatial.distance.jensenshannon(control_probs.values, variation_probs.values, axis=1).mean()
            relative_measures["js_dist"].append(js_dist)
            relative_measures["kl_div"].append(sp.stats.entropy(control_probs.values, variation_probs.values, axis=1).mean())

            # more_than_reversal = js_dist > reversal_dists
            # relative_measures["more_than_reversal"].append(more_than_reversal.mean())
    df = pd.DataFrame.from_dict(relative_measures)
    return df


def get_distances_for_prompt_type(distances):
    return distances.groupby('prompt_type', sort=False).apply(lambda x: x.js_dist.mean(),
                                                              include_groups=False).to_frame().rename(
        columns={0: "mean_distance"})

def get_distances_for_item(distances):
    return distances.groupby('title', sort=False).apply(lambda x: x.js_dist.mean(),
                                                        include_groups=False).to_frame().rename(
        columns={0: "mean_distance"})

def get_item_divergences(divergences):
    return divergences.groupby(['title', 'version'], sort=False, as_index=False).apply(lambda x: x.js_dist.mean(), include_groups=False).rename(
        columns={None: "mean_distance"})

def summarize_missing_probs(df):
    non_options_mask = (df["prompt_type"] != "options") & (df["prompt_type"] != "options_flipped")
    options_mask = (df["prompt_type"] == "options") | (df["prompt_type"] == "options_flipped")
    return pd.concat([
        (df[["Yes_probs", "No_probs"]][non_options_mask] == 0).astype(int).sum(axis=0),
        (df[["A_prob", "B_prob"]][options_mask] == 0).astype(int).sum(axis=0)
    ])

# Modifies DF due to inplace `organize_distribution` call
def analyze_model_divergences(df):
    model_results = organize_distribution(df)
    divergences = calculate_relative_measures(model_results)
    prompt_divergences = get_distances_for_prompt_type(divergences)
    item_divergences = get_distances_for_item(divergences)
    return divergences, prompt_divergences, item_divergences