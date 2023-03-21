from transformers import TFBertForSequenceClassification, BertTokenizerFast
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

question_labels = {
    "tf": 2,
    "mc": 6
}
question_input_file = {
    "tf": 'data/intervalQA/tf_questions.csv',
    "mc": 'data/intervalQA/mc_questions.csv'
}
model_output_file = {
    "tf": 'saved_model/tf_model',
    "mc": 'saved_model/mc_model'
}

model_to_train="tf"

transformer_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                    num_labels=question_labels[model_to_train])  # model used for classificaiton
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',
                                              max_length=256,  # max length of the text that can go to BERT
                                              pad_to_max_length=True)  # pads shorter sequences of text up to the max length

data = pd.read_csv(question_input_file[model_to_train])
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
    learning_rate=5e-5)  # I purposely used a small learning rate for finetuning the model

transformer_model.compile(optimizer=optimizer,
                          loss=transformer_model.hf_compute_loss,
                          metrics=['accuracy'])

transformer_model.fit(train_data.shuffle(1000).batch(16),
                      epochs=1, batch_size=16,
                      validation_data=valid_data.batch(16))

transformer_model.save_pretrained(model_output_file[model_to_train])