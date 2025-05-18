import pandas as pd

from prompts import prompt_types
import numpy as np
import scipy as sp

def get_aff_unaff_columns(prompt_type):
    match prompt_type:
        case "yes_or_no" | "no_or_yes" | "agreement" | "disagreement_negation":
            return "Yes_probs", "No_probs"
        case "disagreement" | "agreement_negation" | "negation":
            return "No_probs", "Yes_probs"
        case "options":
            return "A_probs", "B_probs"
        case "options_flipped":
            return "B_probs", "A_probs"
        case _:
            raise ValueError()


def organize_distribution(model_results):
    model_results["Yes_probs"] = model_results["Yes_probs"].transform(np.exp)
    model_results["No_probs"] = model_results["No_probs"].transform(np.exp)
    model_results["A_probs"] = model_results["A_probs"].transform(np.exp)
    model_results["B_probs"] = model_results["B_probs"].transform(np.exp)
    model_results["Other_prob"] = 1 - model_results["Yes_probs"] - model_results["No_probs"]

    for group, indices in model_results.groupby("prompt_type").indices.items():
        aff_column, unaff_column = get_aff_unaff_columns(group)
        model_results.loc[indices, "Aff_prob"] = model_results[aff_column][indices]
        model_results.loc[indices, "UnAff_prob"] = model_results[unaff_column][indices]

    model_results["entropy"] = model_results[["Aff_prob", "UnAff_prob", "Other_prob"]].apply(lambda x: sp.stats.entropy(x), axis=1)

    return model_results

def calculate_js_distance(model_results):
    divergences = {
        "title": list(),
        "version": list(),
        "prompt_type": list(),
        "js_dist": list(),
        "more_than_reversal": list()
    }
    # control and variation pairs
    for group, df in model_results.groupby(['title', 'version'], sort=False):
        control_mask = df.prompt_type == "yes_or_no"
        control_probs = df[["Aff_prob", "UnAff_prob", "Other_prob"]][control_mask]
        reversal_probs = df[["UnAff_prob", "Aff_prob", "Other_prob"]][control_mask]
        reversal_dists = np.array([sp.spatial.distance.jensenshannon(x, y) for x, y in zip(control_probs.values, reversal_probs.values)])
        for variation in prompt_types[1:]:
            variation_mask = df.prompt_type == variation
            variation_probs = df[["Aff_prob", "UnAff_prob", "Other_prob"]][variation_mask]
            # TODO numpy it
            js_dist = np.array([sp.spatial.distance.jensenshannon(x, y) for x, y in zip(control_probs.values, variation_probs.values)])
            more_than_reversal = js_dist > reversal_dists
            divergences["title"].append(group[0])
            divergences["version"].append(group[1])
            divergences["prompt_type"].append(variation)
            divergences["js_dist"].append(js_dist.mean())
            divergences["more_than_reversal"].append(more_than_reversal.mean())
    df = pd.DataFrame.from_dict(divergences)
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

def analyze_model_divergences(df):
    model_results = organize_distribution(df)
    divergences = calculate_js_distance(model_results)
    prompt_divergences = get_distances_for_prompt_type(divergences)
    item_divergences = get_distances_for_item(divergences)
    return divergences, prompt_divergences, item_divergences