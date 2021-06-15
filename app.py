from flask import Flask
import numpy as np
import re
import tensorflow as tf
from pywebio.input import input,TEXT
from pywebio.output import put_tabs,put_table
from pywebio.platform.flask import webio_view
from pywebio import start_server
import argparse

import joblib 
tags_dict = joblib.load('tags.joblib')
word_tokenizer = joblib.load('tokenizer.joblib')
tflite_model = joblib.load('rnn_lite_model.tflite')

interpreter = tf.lite.Interpreter(model_content = tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details= interpreter.get_output_details()


app = Flask(__name__)

def text_preprocess(text):
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = text.rstrip()
    text = text.lstrip()
    return text

def predict(text, max_len, tags_dict, word_tokenizer):

    original_text = text
    text = text_preprocess(text)
    text = text.split(' ')
    words = word_tokenizer.texts_to_sequences(text)
    words_dict = word_tokenizer.word_index
    words_dict = dict([(value, key) for key, value in words_dict.items()])
    words = np.array(words)
    words = words.ravel()

    words = tf.keras.preprocessing.sequence.pad_sequences(
        [words], maxlen = max_len, padding = 'pre',
        truncating='post', value=0.0 )
  
    words = np.array(words)
    original_words = words
    words = words.astype('float32')
    
    interpreter.set_tensor(input_details[0]['index'], words)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    pred_prob = np.max(pred,axis=2)
    pred = np.argmax(pred,axis=2)

    pred = pred.ravel()
    pred = np.trim_zeros(pred, 'f')
    pred_prob = pred_prob.ravel()
    pred_prob = pred_prob[(pred_prob.shape[0] - pred.shape[0]):]
    original_words = original_words.ravel()
    original_words = np.trim_zeros(original_words, 'f')

    output = []
    for i in range(len(pred)):
        temp = []
        if words_dict[original_words[i]] == '<OOV>':
            temp.append(text[i])
        else:
            temp.append(words_dict[original_words[i]])
        temp.append(tags_dict[pred[i]])
        temp.append(pred_prob[i])
        output.append(temp)

    return original_text, output


def predict_output():
    text = input('Parts Of Speech Tagger',placeholder = "Enter the text", type = TEXT)
    
    max_len = 125
    text, output = predict(text, max_len,tags_dict, word_tokenizer)
    
    put_tabs([
    {'title': 'Results', 'content': [
        put_table(output, header=['Words', 'Tag', 'Probability']) ]},
    {'title': 'Input text', 'content': text}
    ])
    
app.add_url_rule('/tool', 'webio_view', webio_view(predict_output),
            methods=['GET', 'POST', 'OPTIONS'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(predict, port=args.port)
  