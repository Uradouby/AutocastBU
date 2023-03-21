import os
import json
import pickle
import numpy as np
from transformers import TFBertForSequenceClassification, BertTokenizerFast, TextClassificationPipeline
import tensorflow as tf

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',
                                              max_length=256,  # max length of the text that can go to BERT
                                              pad_to_max_length=True)  # pads shorter sequences of text up to the max length

mc_model = TFBertForSequenceClassification.from_pretrained("saved_model/mc_model", num_labels=6)
tf_model = TFBertForSequenceClassification.from_pretrained("saved_model/tf_model", num_labels=2)


def format_output(output, size):

    preds = output

    if size > len(output):
        num = output[len(output)-1]/(size-len(output)+1)
        preds[len(output)-1] = num
        for i in range(size-len(output)):
            preds.append(num)

    elif size < len(output):
        preds = output[0:size]
    sum = 0
    minv = 100

    for i in range(len(preds)):
        if preds[i] < minv:
            minv = preds[i]

    for i in range(len(preds)):
        preds[i] = preds[i]-minv+10
        preds[i]= round(preds[i], 6)
    preds=np.array(preds)

    return preds/preds.sum()


autocast_questions = json.load(open('data/autocast/autocast_questions.json')) # from the Autocast dataset
test_questions = json.load(open('data/autocast/autocast_competition_test_set.json'))
test_ids = [q['id'] for q in test_questions]


def get_answer(question):
    if question['qtype'] == 't/f':
        encoded_input = tokenizer(question["question"], return_tensors='tf')
        output = np.array(tf_model(encoded_input)["logits"]).tolist()[0]
        return format_output(output, 2)

    elif question['qtype'] == 'mc':
        encoded_input = tokenizer(question["question"], return_tensors='tf')
        output = np.array(mc_model(encoded_input)["logits"]).tolist()[0]
        return format_output(output, len(question["choices"]))

    elif question['qtype'] == 'num':
        return 0.5


preds = []

for question in test_questions:
    pred = get_answer(question)
    print(pred)
    preds.append(pred)


if not os.path.exists('submission'):
    os.makedirs('submission')

with open(os.path.join('submission', 'predictions.pkl'), 'wb') as f:
    pickle.dump(preds, f, protocol=2)
