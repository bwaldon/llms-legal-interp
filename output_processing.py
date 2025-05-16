import pandas as pd

from prompts import prompt_types
import numpy as np

def convert_version_to_target(prompt_type, version):
    match prompt_type:
        case "yes_or_no" | "no_or_yes":
            match version:
                case "unambiguous_covered":
                    return "Yes"
                case "unambiguous_uncovered":
                    return "No"
                case _:
                    return None
        case "agreement" | "disagreement_negation":
            match version:
                case "unambiguous_covered":
                    return "Yes"
                case "unambiguous_uncovered":
                    return "No"
                case _:
                    return None
        case "agreement_negation" | "disagreement":
            match version:
                case "unambiguous_covered":
                    return "No"
                case "unambiguous_uncovered":
                    return "Yes"
                case _:
                    return None

        case "options":
            match version:
                case "unambiguous_covered":
                    return "A"
                case "unambiguous_uncovered":
                    return "B"
                case _:
                    return None

        case "options_flipped":
            match version:
                case "unambiguous_covered":
                    return "B"
                case "unambiguous_uncovered":
                    return "A"
                case _:
                    return None
        case _:
            return None









def get_prompt_type_corrects(prompt_type, results):
    pt_results = results[results["prompt_type"] == prompt_type]
    # 1/0 encoding 1 is correct 0 is incorrect
    return pt_results["target"] == pt_results["output"]


def get_prompt_type_results(prompt_type, results):
    pt_results = results[results["prompt_type"] == prompt_type]
    corrects = get_prompt_type_corrects(prompt_type, results)
    return corrects, np.logical_not(corrects), pt_results.output.value_counts()

