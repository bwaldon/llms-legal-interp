import dspy
from typing import Literal
#1 Binary (Meaning) Judgements
# Question style similar to summarization from Blair-Stanek et al.
def binary_option_question(scenario, definition, options):
    return f"""
        Consider the following:
        {scenario} 
        Does it contain an instance of {definition}? Answer with ONLY the string {options[0]} or {options[1]}:   
        """
def no_or_yes_question(scenario, definition):
    return binary_option_question(scenario, definition, ['No', 'Yes'])

def yes_or_no_question(scenario, definition):
    return binary_option_question(scenario, definition, ['Yes', 'No'])

def make_it_do_you_agree(item, actor = "Insurance Company"):
    return f"""{item.header}
    {item.continuation}
    {actor} decided that the claim is {item.version} under the policy.
    Do you agree?
    """



class YesOrNoPrompt(dspy.Signature):
    """Make it a classification Yes, or No question."""

    question: str = dspy.InputField()
    sentiment: Literal['Yes', 'No'] = dspy.OutputField()

if __name__ == '__main__':
    print(yes_or_no_question(
        "Gavin works as an A/C repair technician in a small town. One day, Gavin is hired to repair an air conditioner located on the second story of a building. Because Gavin is an experienced repairman, he knows that the safest way to access the unit is with a sturdy ladder. While climbing the ladder, Gavin loses his balance and falls, causing significant injury. Because of this, he subsequently has to stop working for weeks. Gavin files a claim with his insurance company for lost income.",
        "missed employment due to injuries that occur under regular working conditions."
    ))

