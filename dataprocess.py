import json
import copy
import random
import pandas as pd


def questions_trans_to_fid(autocast_questions):
    fid_questions=[]
    for q in autocast_questions:
        if q['answer'] is None:
            continue
        fid_questions.append(
            {
                "question_id": str(q['id']),
                "question": q['question'],
                "answers": [str(q['answer'])],
                'target': str(q['answer']),
                'ctxs': [
                    {
                        "title": "",
                        "text": ""
                    }
                ]
            }
        )

    out_file = "data/autocast/train_data.json"
    with open(out_file, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(fid_questions, indent=4, ensure_ascii=False) + "\n")
    with open("data/autocast/eval_data.json", "w", encoding='utf-8') as writer:
        writer.write(json.dumps(fid_questions, indent=4, ensure_ascii=False) + "\n")


def get_questions():
    questions = json.load(open('data/autocast/autocast_questions.json'))
    negations = json.load(open('data/autocast/negated_tf_questions.json'))
    qid_to_negation = get_question_dict(negations)
    balanced_questions = []
    for q in questions:
        if q['qtype'] == 't/f':
            assert q['question'] == qid_to_negation[q['id']]['original']
            negated_q = copy.deepcopy(q)
            negated_q['id'] = 'N'+negated_q['id']
            negated_q['question'] = qid_to_negation[q['id']]['negated']
            for t in negated_q['crowd']:
                t['forecast'] = 1 - t['forecast']  # flip the forecast probabilities
            if q['answer'] is not None:  # flip the resolution
                if q['answer'] == 'yes':
                    negated_q['answer'] = 'no'
                else:
                    negated_q['answer'] = 'yes'
            balanced_questions.append(negated_q)
        balanced_questions.append(q)
    return balanced_questions


def split_questions_by_type(autocast_questions):
    mc_questions = []
    num_questions = []
    tf_questions = []

    for q in autocast_questions:
        if q['qtype'] == 'mc':
            mc_questions.append(q)
        elif q['qtype'] == 'num':
            num_questions.append(q)
        elif q['qtype'] == 't/f':
            tf_questions.append(q)
    return mc_questions, num_questions, tf_questions


def split_questions_by_status(autocast_questions, time):
    resolved_before_time_questions = []
    resolved_after_time_questions = []
    unresolved_before_time_questions = []
    unresolved_after_time_questions = []
    for q in autocast_questions:
        if q['status'] == 'Resolved':
            if q['close_time'] < time:
                resolved_before_time_questions.append(q)
            else:
                resolved_after_time_questions.append(q)
        else:
            if q['publish_time'] < time:
                unresolved_before_time_questions.append(q)
            else:
                unresolved_after_time_questions.append(q)
    return resolved_before_time_questions, resolved_after_time_questions, unresolved_before_time_questions, unresolved_after_time_questions


def get_question_dict(questions):
    qid_to_question = {q['id']: q for q in questions}
    return qid_to_question

def data_process_num(num_qs,bin_num):
    datas = []
    labels = []
    if bin_num < 0:
        bin_num = 1
    std = 1/bin_num
    bins=[]
    for i in range(bin_num):
        bins.append(i*std)
    bins.append(1)
    print(bins)
    for q in num_qs:
        if q['answer'] is None:
            continue
        for i in range(bin_num+1):
            if q['answer']<=bins[i]:
                labels.append(i)
                datas.append(str(q['question'])+"\n")
                break
    return pd.DataFrame(data={'question': datas, 'label': labels})


def data_augment_mc(mc_qs):
    datas = []
    labels = []
    augment_rate = 3
    for q in mc_qs:
        if q['answer'] is None:
            continue
        ans = ord(q['answer']) - ord('A')
        tmp = q['question'] + '\n'
        for i in range(0, len(q['choices'])):
            tmp = tmp + str(i) + '.' + q['choices'][i] + '\n'
        datas.append(tmp)
        labels.append(min(ans, 5))
        # augment data
        for _ in range(augment_rate):
            shuffles = [i for i in range(0, len(q['choices']))]
            while len(shuffles) > 1 and shuffles[ans] == ans:
                random.shuffle(shuffles)
            tmp = q['question'] + '\n'
            for i in range(0, len(q['choices'])):
                tmp = tmp + str(i) + '.' + q['choices'][shuffles[i]] + '\n'
            datas.append(tmp)
            tmplabel = 5
            for i in range(len(shuffles)):
                if shuffles[i] == ans:
                    tmplabel = min(i, tmplabel)
            labels.append(tmplabel)

    return pd.DataFrame(data={'question': datas, 'label': labels})

questions = get_questions()
mc_qs, num_qs, tf_qs = split_questions_by_type(questions)
print(len(mc_qs))
print(len(num_qs))
print(len(tf_qs))
qid_to_question = get_question_dict(questions)

#questions_trans_to_fid(num_qs)

df = pd.DataFrame(data={'question': [q['question'] for q in tf_qs], 'label': [1 if q['answer'] =='yes' else 0 for q in tf_qs]})
df.to_csv("data/intervalQA/tf_questions.csv", index=False)
df = data_augment_mc(mc_qs)
df.to_csv("data/intervalQA/mc_questions.csv", index=False)

bin_num=20  # when processing num questions we want to dicreet the continuous answer by using bins
df = data_process_num(num_qs, bin_num=bin_num)
df.to_csv("data/intervalQA/num_questions_"+str(bin_num)+".csv", index=False)

#print(qid_to_question.keys())
