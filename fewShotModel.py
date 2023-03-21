
import openai
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import pairwise

openai.api_key="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"
codex_engine = "code-davinci-002"
few_shot_max_tokens = 256
engine_temperature = 0
engine_topP = 0

class fewShotModel():

    def __init__(self, questions):
        self.questions = questions
        self.input = []
        self.output = []

    def train(self):
        input=''
        for i in range(1, 3):
            q=self.questions[i]
            input=input+str(q['question'])+str(q['answer'])+'\n\n'
        input=input+str(self.questions[0]['question'])
        print(input)
        few_shot_output = openai.Completion.create(engine=codex_engine,
                                                   prompt=input,
                                                   max_tokens=few_shot_max_tokens,
                                                   temperature=engine_temperature,
                                                   top_p=engine_topP)['choices'][0]['text']
        print(few_shot_output)
