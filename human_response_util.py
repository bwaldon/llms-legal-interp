import pandas as pd
import numpy as np

if __name__ ==  "__main__":
    yes_responses = np.sum(human_responses["individual_judgment"] == "yes")
    no_responses = np.sum(human_responses["individual_judgment"] == "no")
    print(f"Yes proportion : {yes_responses/ (yes_responses + no_responses) }")
    print(f"No proportion : {no_responses / (yes_responses + no_responses)}")
