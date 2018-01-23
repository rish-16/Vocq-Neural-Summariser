# export LC_ALL=en_US.UTF-8
# export LANG=en_US.UTF-8

import dill as dpickle
import pandas as pd
import numpy as np
import logging
import glob

from sklearn.model_selection import train_test_split
from seq2seq_utils import load_decoder_inputs, load_encoder_inputs, load_text_processor
from seq2seq_utils import viz_model_architecture
from seq2seq_utils import Seq2Seq_Inference

from ktext.preprocess import processor
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, BatchNormalization
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint

pd.set_option('display.max_colwidth', 500)
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

df = pd.read_csv('github_issues.csv', encoding='utf-8').sample(n=2000000)
train_df, test_df = train_test_split(df, test_size=0.10)

print ('Train: {} rows | {} columns'.format(train_df.shape[0], train_df.shape[1]))
print ('Test: {} rows | {} columns'.format(test_df.shape[0], test_df.shape[1]))

train_body_raw = train_df.body.tolist()
train_title_raw = train_df.issue_title.tolist()

body_pp = processor(keep_n=8000, padding_maxlen=70)
title_pp = processor(append_indicators=True, keep_n=4500, padding_maxlen=12, padding ='post')

train_body_vecs = body_pp.fit_transform(train_body_raw)
train_title_vecs = title_pp.fit_transform(train_title_raw)

print ('Original: \n{}\n\n'.format(train_body_raw[0]))
print ('Vectors: \n{}\n\n'.format(train_body_vecs[0]))

with open('body_pp.dpkl', 'wb') as f:
	dpickle.dump(body_pp, f)

with open('title_pp.dpkl', 'wb') as f:
	dpickle.dump(title_pp, f)

np.save('train_title_vecs.npy', train_title_vecs)
np.save('train_body_vecs.npy', train_body_vecs)

encoder_input_data, doc_length = load_encoder_inputs('train_body_vecs.npy')
decoder_input_data, decoder_target_data = load_decoder_inputs('train_title_vecs.npy')

num_encoder_tokens, body_pp = load_text_processor('body_pp.dpkl')
num_decoder_tokens, title_pp = load_text_processor('title_pp.dpkl')

latent_dim = 300

encoder_inputs = Input(shape=(doc_length), name='Encoder_Input')
x = Embedding(num_encoder_tokens, latent_dim, name='Body_Word_Embedding', mask_zero=False)(encoder_inputs)
x = BatchNormalization(name='Encoder_BN_1')(x)
_, state_h = GRU(latent_dim, return_state=True, name='Encoder_Last_GRU')(x)
encoder_model = Model(inputs=encoder_inputs, ouputs=state_h, name='Encoder_Model')
seq2seq_encoder_out = encoder_model(encoder_inputs)

decoder_inputs = Input(shape=(None,), name='Decoder_Input')
dec_emb = Embedding(num_decoder_tokens, latent_dim, name='Decoder_Word_Embedding', mask_zero=False)(decoder_inputs)
dec_bn = BatchNormalization(name='Decoder_BN_1')(dec_emb)
decoder_gru = GRU(latent_dim, return_state=True, return_sequences=True, name='Decoder_GRU')
decoder_gru_output, _ = decoder_gru(dec_bn, initial_state=seq2seq_encoder_out)
x = BatchNormalization(name='Decoder_BN_2')(decoder_gru_output)
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='Final_Output_Dense')
decoder_outputs = decoder_dense(x)

seq2seq_Model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')
seq2seq_Model.summary()
viz_model_architecture(seq2seq_Model)

script_name_base = 'seq2seq'
csv_logger = CSVLogger('{:}.log'.format(script_name_base))
model_checkpoint = ModelCheckpoint('{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format(script_name_base), save_best_only=True)

batch_size = 1200
epochs = 7

history = seq2seq_Model.fit([encoder_input_data, decoder_input_data], np.expand_dims(decoder_target_data, -1), batch_size=batch_size, epochs=epochs, validation_split=0.12, callbacks=[csv_logger, model_checkpoint])
seq2seq_Model.save('seq2seq_model_tutorial.h5')

seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=body_pp, decoder_preprocessor=title_pp, seq2seq_model=seq2seq_Model)
seq2seq_inf.demo_model_predictions(n=50, issue_df=testdf)
