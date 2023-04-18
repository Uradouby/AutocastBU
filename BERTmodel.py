from transformers import TFBertForSequenceClassification, BertTokenizerFast
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

model_name = "bert-base-cased"  # the bert model you want to use
train_question_type = "num10"  #The type of questions to train
lr = 5e-5  #learning rate
batch_size = 16  #batch size

question_labels = {
    "tf": 2,
    "mc": 6,
    "num5": 6,
    "num10": 11,
    "num20": 21
}

question_input_file = {
    "tf": 'data/intervalQA/tf_questions.csv',
    "mc": 'data/intervalQA/mc_questions.csv',
    "num5": 'data/intervalQA/num_questions_5.csv',
    "num10": 'data/intervalQA/num_questions_10.csv',
    "num20": 'data/intervalQA/num_questions_20.csv'
}

model_output_file = {
    "tf": 'saved_model/'+model_name+'/tf_model',
    "mc": 'saved_model/'+model_name+'/mc_model',
    "num5": 'saved_model/'+model_name+'/num5_model',
    "num10": 'saved_model/'+model_name+'/num10_model',
    "num20": 'saved_model/'+model_name+'/num20_model',
}


transformer_model = TFBertForSequenceClassification.from_pretrained(model_name,
                                                                    num_labels=question_labels[train_question_type])  # model used for classificaiton
tokenizer = BertTokenizerFast.from_pretrained(model_name,
                                              max_length=256,  # max length of the text that can go to BERT
                                              pad_to_max_length=True)  # pads shorter sequences of text up to the max length

data = pd.read_csv(question_input_file[train_question_type])
data.dropna(inplace=True)

X = data['question']
y = data['label']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_encoded = dict(tokenizer(list(X_train.values),
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=256,  # max length of the text that can go to BERT
                                 pad_to_max_length=True,  # add [PAD] tokens
                                 return_attention_mask=True))

X_valid_encoded = dict(tokenizer(list(X_valid.values),
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=256,  # max length of the text that can go to BERT
                                 pad_to_max_length=True,  # add [PAD] tokens
                                 return_attention_mask=True))

train_data = tf.data.Dataset.from_tensor_slices((X_train_encoded, list(y_train.values)))
valid_data = tf.data.Dataset.from_tensor_slices((X_valid_encoded, list(y_valid.values)))

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr)  # I purposely used a small learning rate for finetuning the model

transformer_model.compile(optimizer=optimizer,
                          loss=transformer_model.hf_compute_loss,
                          metrics=['accuracy'])

transformer_model.fit(train_data.shuffle(1000).batch(batch_size),
                      epochs=1, batch_size=batch_size,
                      validation_data=valid_data.batch(batch_size))

transformer_model.save_pretrained(model_output_file[train_question_type])