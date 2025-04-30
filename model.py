from vllm import LLM, SamplingParams
import csv
from typing import List

class MetaLinguisticPrompt:
    def __init__(self, nudge, topic="landscaping"):
        self.nudge = nudge
        self.topic = topic

        if topic == "landscaping":
            self.text = ('Can you tell me the ordinary meaning of "landscaping",'
                         ' and whether installing an in-ground trampoline would be included in such meaning?')
        elif topic == "restraining":
            self.text = ('Can you tell me the ordinary meaning of "physically restraining someone",'
                         ' and whether holding someone at gunpoint would be included in such meaning?')
        elif topic == "vague":
            self.text: str = self.from_vague_contracts()

        self.add_nudge()


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
        #todo: implement
        with open("vague_contracts/data/1_falseconsensus/demo-merged.csv") as f:
            r = csv.reader(f, delimiter=",", quotechar='"')
            data = [row for row in r]

        context, policy, ambiguity = data[3], data[4], data[11]

        return ""

class JudgementModel(LLM):
    def __init__(self, model_name, max_model_len=1024):
        self.model_name = model_name
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
        self.max_model_len = max_model_len
        self.llm = LLM(model_name, max_model_len=max_model_len)

    def infer(self, prompts: List[MetaLinguisticPrompt]) -> List[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]


