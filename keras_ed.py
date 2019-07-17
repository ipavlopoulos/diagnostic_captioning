# -*- coding: utf-8 -*-

"""
MIC Encoder Decoder in Keras
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import random
import os
import json
import numpy as np
import pandas as pd
import pickle
#import re
#import time
#from glob import glob
#from PIL import Image
#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle


from os import listdir
from pickle import load
from numpy import array
from numpy import argmax
from pandas import DataFrame
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

import keras

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GRU, LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.layers.pooling import GlobalMaxPooling2D

from keras.preprocessing.text import Tokenizer

def preprocess(text, start="startsequence", end="endsequence"):
    text = text.lower()
    return start + ' ' + text + ' ' + end

def encode(img_path):
    image = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = keras.preprocessing.image.img_to_array(image)
    # prepare the image for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = keras.applications.densenet.preprocess_input(image)
    return encoder.predict(image, verbose=0)


def create_sequences(tokenizer, caption, image1, image2, max_length):
    Ximages1, Ximages2, XSeq, y = list(), list(), list(), list()
    vocab_size = len(tokenizer.word_index) + 1
    # integer encode the description
    seq = tokenizer.texts_to_sequences([caption])[0]
    # split one sequence into multiple X,y pairs
    for i in range(1, len(seq)):
        # select
        in_seq, out_seq = seq[:i], seq[i]
        # pad input sequence
        in_seq = keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length)[0]
        # encode output sequence
        out_seq = keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
        # store
        Ximages1.append(image1)
        Ximages2.append(image2)
        XSeq.append(in_seq)
        y.append(out_seq)
    return [Ximages1, Ximages2, XSeq, y]

# define the captioning model
def define_model(vocab_size, max_length, loss="categorical_crossentropy"):
    # feature extractor (encoder)
    inputs1 = Input(shape=(7, 7, 512))
    inputs2 = Input(shape=(7, 7, 512))
    fe1 = GlobalMaxPooling2D()(inputs1)
    fe2 = GlobalMaxPooling2D()(inputs2)
    fe3 = Dense(128, activation='relu')(fe1)
    fe4 = Dense(128, activation='relu')(fe2)
    fe5 = concatenate([fe3, fe4])
    fe6 = RepeatVector(max_length)(fe5)
    # embedding
    inputs3 = Input(shape=(max_length,))
    emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs3)
    emb3 = GRU(128, return_sequences=True)(emb2)
    emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
    # merge inputs
    merged = concatenate([fe6, emb4])
    # language model (decoder)
    lm2 = GRU(256)(merged)
    lm3 = Dense(128, activation='relu')(lm2)
    outputs = Dense(vocab_size, activation='softmax')(lm3)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    # loss could also be e.g. nltk.translate.bleu_score.sentence_bleu
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='plot.png')
    return model


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photos, max_length):
    # seed the generation process
    in_text = start
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photos[0], photos[1], sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == end:
            break
    return in_text


# generate a description for an image
def generate_desc_beam(model, tokenizer, photos, max_length, beam_size=10):
    in_text = [start]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        tmp = []
        for s in start_word:
            sequence = tokenizer.texts_to_sequences(s[0])
            sequence = keras.preprocessing.sequence.pad_sequences([sequence[0]], maxlen=max_length)
            yhat = model.predict([photos[0], photos[1], np.array(sequence)], verbose=0)
            word_yhat = np.argsort(yhat[0])[-beam_size:]
            for w in word_yhat:
                nextcap, prob = s[0], s[1]
                nextcap += ' ' + word_for_id(w, tokenizer)
                prob += yhat[0][w]
                # print (nextcap, prob)
                tmp.append([nextcap, prob])
        start_word = tmp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_size:]
    start_word = start_word[-1][0]
    intermediate_caption = start_word
    final_caption = []
    for i in intermediate_caption.split():
        if i != end:
            final_caption.append(i)
        else:
            break
    final_caption = ' '.join(final_caption[1:])
    return final_caption


# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key][0], photos[key][1], max_length)
        # store actual and predicted
        actual.append([desc.split()])
        predicted.append(yhat.split())
    # calculate BLEU score
    bleu = corpus_bleu(actual, predicted)
    return bleu


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(captions, image_tuples, tokenizer, max_length, n_step, validation=False, validation_num=None):
    while 1:
        # iterate over patients - hold some for validation
        patients = list(captions.keys())
        if validation:
            assert validation_num > 0
            patients = patients[-validation_num:]
        elif not validation:
            if validation_num > 0:
                patients = patients[:-validation_num]

        for i in range(0, len(patients), n_step):
            Ximages1, Ximages2, XSeq, y = list(), list(), list(), list()
            for j in range(i, min(len(patients), i + n_step)):
                patient_id = patients[j]
                # retrieve text input
                caption = captions[patient_id]
                # generate input-output pairs (many images in each batch)
                img1 = image_tuples[patient_id][0][0]
                img2 = image_tuples[patient_id][1][0]
                in_img1, in_img2, in_seq, out_word = create_sequences(tokenizer, caption, img1, img2, max_length)
                for k in range(len(in_img1)):
                    Ximages1.append(in_img1[k])
                    Ximages2.append(in_img2[k])
                    XSeq.append(in_seq[k])
                    y.append(out_word[k])
                # yield this batch of samples to the model
                # print (array(Ximages1).shape)
            yield [array(Ximages1), array(Ximages2), array(XSeq)], array(y)


# evaluate the skill of the model
def evaluate_n_visualise(model, descriptions, photos, tokenizer, max_length, size=5, beam=0):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc in descriptions.items():
        if random.random() > 0.8:
            # generate description
            if beam > 0:
                yhat = generate_desc_beam(model, tokenizer, photos[key], max_length, beam_size=beam)
            else:
                yhat = generate_desc(model, tokenizer, photos[key], max_length)
            # store actual and predicted
            print('Actual:    %s' % desc)
            print('Predicted: %s\n' % yhat)
            actual.append([desc.split()])
            predicted.append(yhat.split())
            if len(actual) >= size: break
    # calculate BLEU score
    bleu = corpus_bleu(actual, predicted)
    return bleu



DATA_PATH = "iu_xray/"
DATA_IMAGES_PATH = os.path.join(DATA_PATH, "iu_xray_images")

iuxray_major_tags = json.load(open(DATA_PATH + "iu_xray_major_tags.json"))
iuxray_auto_tags = json.load(open(DATA_PATH + "iu_xray_auto_tags.json"))
iuxray_captions = json.load(open(DATA_PATH + "iu_xray_captions.json"))

iuxray_ids = list(iuxray_captions.keys())
iuxray_caption_texts = [iuxray_captions[iid].split() for iid in iuxray_ids]
iuxray_captions_num = len(iuxray_ids)
print("# texts: {0}".format(iuxray_captions_num))
raw_texts = [" ".join(t) for t in iuxray_caption_texts]
print(len(raw_texts), len(set(raw_texts)))

patient_images = {}
for visit in iuxray_ids:
    patient = visit[3:].split("_")[0]
    if patient in patient_images:
        patient_images[patient].append(visit)
    else:
        patient_images[patient] = [visit]

# a model should be reading both images per patient while excluding other patients
iuxray_ids_img1 = [patient_images[patient][0] for patient in patient_images if len(patient_images[patient]) == 1]
iuxray_ids_img2 = [patient_images[patient][0] for patient in patient_images if len(patient_images[patient]) == 2]
iuxray_ids_img3 = [patient_images[patient][0] for patient in patient_images if len(patient_images[patient]) > 2]
print(len(iuxray_ids_img1), len(iuxray_ids_img2), len(iuxray_ids_img3))
print(patient_images)

train_path = os.path.join(DATA_PATH, "train_images.tsv")
test_path = os.path.join(DATA_PATH, "test_images.tsv")
train_data = pd.read_csv(train_path, sep="\t", header=None)
test_data = pd.read_csv(test_path, sep="\t", header=None)
train_data.columns = ["id", "caption"]
test_data.columns = ["id", "caption"]

start, end = "startsequence", "endsequence"


image_captions = dict(
    zip(train_data.id.to_list() + test_data.id.to_list(), train_data.caption.to_list() + test_data.caption.to_list()))
print(len(image_captions))
patient_captions = {patient: [image_captions[img] for img in patient_images[patient]] for patient in patient_images}
print(len(patient_captions))
# discard patients without both XRays
ids = list(patient_captions.keys())
for patient in ids:
    if len(patient_captions[patient]) != 2:
        del patient_captions[patient], patient_images[patient]
    else:
        patient_captions[patient] = preprocess(patient_captions[patient][0])
        patient_images[patient] = [os.path.join(DATA_IMAGES_PATH, img_name) for img_name in patient_images[patient]]

ids = list(patient_captions.keys())
random.shuffle(ids)
sample_size = int(len(ids) * .1)
train_ids = ids[:-sample_size]
test_ids = ids[-sample_size:]
print(len(train_ids), len(test_ids))


in_layer = Input(shape=(224, 224, 3))
encoder = VGG16(include_top=False, input_tensor=in_layer)
print(encoder.summary())


try:
    patient_images = pickle.load(open("patient_images.pkl", "rb"))
    patient_captions = pickle.load(open("patient_captions.pkl", "rb"))
    train_encoded_images = pickle.load(open("train_encoded_images.pkl", "rb"))
except:
    from tqdm import tqdm
    train_encoded_images = {}
    for pid in tqdm(train_ids):
        train_encoded_images[pid] = [encode(img_path) for img_path in patient_images[pid]]
    train_captions = {pid: patient_captions[pid] for pid in train_ids}


train_texts = [patient_captions[pid] for pid in train_ids]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


print(np.array(create_sequences(tokenizer, train_texts[0], train_encoded_images[train_ids[0]][0][0],
                                train_encoded_images[train_ids[0]][1][0], 58)).shape)



#bleu = evaluate_n_visualise(loaded_model, test_captions, test_encoded_images, tokenizer, max_length, beam=10)


test_encoded_images = {}
for pid in tqdm(test_ids):
    test_encoded_images[pid] = [encode(img_path) for img_path in patient_images[pid]]

test_captions = {pid: patient_captions[pid] for pid in test_ids}

# define experiment
verbose = 1
n_epochs = 300
max_length = 60
n_patients_per_update = 16
val_len = int(.01 * len(train_ids))
train_len = len(train_ids) - val_len
train_steps = int(train_len / n_patients_per_update)
val_steps = int(val_len / n_patients_per_update)
model_name = 'show_n_tell.e' + str(n_epochs) + '.ml' + str(max_length) + '.ppu' + str(
    n_patients_per_update) + '.val' + str(val_steps)
print(train_steps, val_steps, model_name)

show_n_tell = define_model(vocab_size, max_length, loss="categorical_crossentropy")

# Train
train_gen = data_generator(train_captions, train_encoded_images, tokenizer, max_length, n_patients_per_update,
                           validation=False, validation_num=val_len)
val_gen = data_generator(train_captions, train_encoded_images, tokenizer, max_length, n_patients_per_update,
                         validation=True, validation_num=val_len)
early = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto',
                                      restore_best_weights=True)
show_n_tell.fit_generator(train_gen, validation_data=val_gen, steps_per_epoch=train_steps, validation_steps=val_steps,
                          epochs=n_epochs, verbose=verbose, callbacks=[early])

# Save & download
show_n_tell.save(model_name + '.h5')

# Load & evaluate
print("Loading...")
loaded_model = keras.models.load_model(model_name + '.h5')
loaded_model.summary()
plot_model(loaded_model, show_shapes=True, to_file=model_name + '.png')
print("Evaluating...")
bleu = evaluate_n_visualise(loaded_model, test_captions, test_encoded_images, tokenizer, max_length, beam=2)
print(bleu)

# See what the model produces in each epoch:
show_n_tell = define_model(vocab_size, max_length, loss="categorical_crossentropy")
for i in range(n_epochs):
    print("### Running epoch ", i, " ###")
    train_gen = data_generator(train_captions, train_encoded_images, tokenizer, max_length, n_patients_per_update,
                               validation=False, validation_num=val_len)
    val_gen = data_generator(train_captions, train_encoded_images, tokenizer, max_length, n_patients_per_update,
                             validation=True, validation_num=val_len)
    show_n_tell.fit_generator(train_gen, validation_data=val_gen, steps_per_epoch=train_steps,
                              validation_steps=val_steps, epochs=1, verbose=verbose)
    for _ in range(3):
        test_id = random.choice(test_ids)
        generation = generate_desc_beam(show_n_tell, tokenizer, test_encoded_images[test_id], max_length)
        print("Predicted:", generation)
        print("Actual   :", test_captions[test_id])
        print()


# the code is based on https://machinelearningmastery.com/develop-a-caption-generation-model-in-keras/
# add visual/semantic attention
# use the first words only
# scale the architecture hierarchicaly (or consecutive)