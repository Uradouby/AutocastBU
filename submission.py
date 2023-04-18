import os
import json
import pickle
import random

import numpy as np
from transformers import TFBertForSequenceClassification, BertTokenizerFast, TextClassificationPipeline
import tensorflow as tf
ENSEMBLE=False
model_name="bert-base-uncased"
num_bins = 5
bins = [i / num_bins for i in range(num_bins+1)]
if ENSEMBLE:
    ensemble_tf = ["bert-base-cased", "bert-large-cased"]
    ensemble_mc = ["bert-base-cased", "bert-large-uncased"]
    ensemble_num = ["bert-base-uncased", "bert-large-cased"]

    ensemble_tf_weights = [0.6, 0.4]
    ensemble_mc_weights = [0.3, 0.7]
    ensemble_num_weights = [0.5, 0.5]

    tf_models = [
        TFBertForSequenceClassification.from_pretrained('saved_model/' + model_name + '/tf_model', num_labels=2) for
        model_name in ensemble_tf]
    mc_models = [
        TFBertForSequenceClassification.from_pretrained('saved_model/' + model_name + '/mc_model', num_labels=6) for
        model_name in ensemble_mc]
    num_models = [
        TFBertForSequenceClassification.from_pretrained('saved_model/' + model_name + '/num' + str(num_bins) + '_model',
                                                        num_labels=num_bins + 1) for model_name in ensemble_num]

else:
    problem_id = ["M6091", "M5927", "M6361", "M8310"]
    mc_model = TFBertForSequenceClassification.from_pretrained('saved_model/' + model_name + '/mc_model', num_labels=6)
    tf_model = TFBertForSequenceClassification.from_pretrained('saved_model/' + model_name + '/tf_model', num_labels=2)
    num_model = TFBertForSequenceClassification.from_pretrained(
        'saved_model/' + model_name + '/num' + str(num_bins) + '_model', num_labels=num_bins + 1)


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


def get_ensemble_answer(question):
    tokenizer = BertTokenizerFast.from_pretrained(model_name,
                                                  max_length=256,  # max length of the text that can go to BERT
                                                  pad_to_max_length=True)  # pads shorter sequences of text up to the max length

    if question['qtype'] == 't/f':
        encoded_input = tokenizer(question["question"], return_tensors='tf')
        tmp_output = [np.array(model(encoded_input)["logits"]).tolist()[0] for model in tf_models]
        output=np.zeros_like(tmp_output[0])
        for i in range(len(tmp_output)):
            output += ensemble_tf_weights[i]*np.array(tmp_output[i])
        return format_output(output, 2)

    elif question['qtype'] == 'mc':
        encoded_input = tokenizer(question["question"], return_tensors='tf')
        tmp_output = [np.array(model(encoded_input)["logits"]).tolist()[0] for model in mc_models]
        output = np.zeros_like(tmp_output[0])
        for i in range(len(tmp_output)):
            output += ensemble_mc_weights[i]*np.array(tmp_output[i])
        return format_output(output, len(question["choices"]))

    elif question['qtype'] == 'num':
        encoded_input = tokenizer(question["question"], return_tensors='tf')
        output = np.array(num_model(encoded_input)["logits"]).tolist()[0]
        return output


def get_answer(question):

    tokenizer = BertTokenizerFast.from_pretrained(model_name,
                                                  max_length=256,  # max length of the text that can go to BERT
                                                  pad_to_max_length=True)  # pads shorter sequences of text up to the max length
    if question['id'] in problem_id:
        print(tokenizer(question["question"], return_tensors='tf'))
        return 0.4
    print(question['id'])

    if question['qtype'] == 't/f':
        encoded_input = tokenizer(question["question"], return_tensors='tf')
        output = np.array(tf_model(encoded_input)["logits"]).tolist()[0]
        return format_output(output, 2)

    elif question['qtype'] == 'mc':
        encoded_input = tokenizer(question["question"], return_tensors='tf')
        output = np.array(mc_model(encoded_input)["logits"]).tolist()[0]
        return format_output(output, len(question["choices"]))

    elif question['qtype'] == 'num':
        encoded_input = tokenizer(question["question"], return_tensors='tf')
        tmp_output = np.array(num_model(encoded_input)["logits"]).tolist()[0]
        output = 0
        for i in range(num_bins):
            output += tmp_output[i]*bins[i]
        return output

preds = []
random.seed(42)

for question in test_questions:
    if ENSEMBLE:
        pred = get_ensemble_answer(question)
    else:
        pred = get_answer(question)
    print(pred)

    preds.append(pred)


if not os.path.exists('submission'):
    os.makedirs('submission')

with open(os.path.join('submission', 'predictions.pkl'), 'wb') as f:
    pickle.dump(preds, f, protocol=2)
