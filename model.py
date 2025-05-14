from vllm import LLM, SamplingParams, CompletionOutput
import csv
from typing import List

class MetaLinguisticPrompt:
    def __init__(self, nudge=None, features=None, topic="landscaping"):
        self.nudge = nudge
        self.features = features
        self.topic = topic

        self.text = ""
        self.get_base_text()

        if features is not None:
            self.add_feature()

        if nudge is not None:
            self.add_nudge()

    def get_base_text(self):
        if self.topic == "landscaping":
            if "question" in self.features:
                self.text += ('Can you consider the ordinary meaning of "landscaping",'
                         ' and decide whether installing an in-ground trampoline would be included in such meaning?')
            elif "arg" in self.features:
                self.text += ('Considering the ordinary meaning of "landscaping", '
                             'installing an in-ground trampoline is landscaping.')
            elif "neg_arg" in self.features:
                self.text += ('Considering the ordinary meaning of "landscaping", '
                             'installing an in-ground trampoline is not landscaping.')
            elif "event" in self.features:
                self.text += ('Consider the ordinary meaning of "landscaping". '
                             'Abhishek was installing an in-ground trampoline in his backyard, when he suffered an injury.')
        elif self.topic == "restraining":
            if "question" in self.features:
                self.text += ('Can you tell me the ordinary meaning of "physically restraining someone",'
                             ' and whether holding someone at gunpoint and telling them not to move would be included in such meaning?')
            elif "arg" in self.features:
                self.text += ('Considering the ordinary meaning of "physically restraining someone", '
                             'holding someone at gunpoint and telling them not to move is landscaping.')
            elif "neg_arg" in self.features:
                self.text += ('Considering the ordinary meaning of "physically restraining someone", '
                             'holding someone at gunpoint and telling them not to move is not landscaping.')
            elif "event" in self.features:
                self.text += ('Consider the ordinary meaning of "physically restraining someone". '
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
            self.text += " Which is more likely? Options: A. His injury is covered. B. His injury is not covered."
        elif "mc_prompt_reverse" in self.features:
            self.text += " Which is more likely? Options: A. His injury is not covered. B. His injury is covered."

        answer_trigger = "\nThe final answer is: "
        self.text += answer_trigger


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
        with open("vague_contracts/data/1_falseconsensus/demo-merged.csv") as f:
            r = csv.reader(f, delimiter=",", quotechar='"')
            data = [row for row in r]

        context, policy, ambiguity = data[3], data[4], data[11]

        return ""

class MetaLinguisticJudgement:
    def __init__(self, model_name, seed, max_model_len=256):
        self.model_name = model_name
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256, seed=seed, prompt_logprobs=10, logprobs=10)
        self.max_model_len = max_model_len
        self.llm = LLM(
            model_name,
            max_model_len=max_model_len,
            seed=seed,
            dtype="float16",
            gpu_memory_utilization=0.95,
            max_num_seqs=8,
            swap_space=8,
        )

    def infer(self, prompts: List[MetaLinguisticPrompt]) -> List[CompletionOutput]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0] for output in outputs]

    def probs(self, prompts: List[MetaLinguisticPrompt]) -> List:
        outputs = self.llm.generate([p.text for p in prompts], self.sampling_params)
        return [output.outputs[0].logprobs for output in outputs]
