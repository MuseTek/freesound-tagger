'''
This file provides the functionality to predict a tag given an audio file.
'''

import librosa
from mel_and_pca_model_funcs import create_mel_and_pca_model
import numpy as np
import pandas as pd
import pdb
import os
from summary_feats_funcs import all_feats_np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pca = pickle.load(open("pca.p","rb"))
sc = pickle.load(open("standard_scaler.p","rb"))

def get_mel_spec(filename,sample_rate=44100):
    '''
    Converts and trims a wav file to a mel spectrogram so that 
    it can be fed into the model.
    '''
    x, sr = librosa.load(filename)
    x = librosa.effects.trim(x)[0]
    wav = librosa.resample(x, 44100, 22050)
    melspec = librosa.feature.melspectrogram(wav,
                                                sr=22050,
                                                n_fft=1764,
                                                hop_length=220,
                                                n_mels=64)
    logmel = librosa.core.power_to_db(melspec)

    return logmel

def get_mel_and_pca(filename, sample_rate=44100):
    '''
    returns both a melgram spectrogram and the pca features for a given audio file
    '''
    x, sr = librosa.load(filename)
    x = librosa.effects.trim(x)[0]
    wav = librosa.resample(x, 44100,22050)
    melspec = librosa.feature.melspectrogram(wav, sr=22050, n_fft=1764,hop_length=220,n_mels=64)
    logmel = librosa.core.power_to_db(melspec)
    logmel = np.reshape(logmel,(1,logmel.shape[0],logmel.shape[1],1))
    x_feats = all_feats_np(x)
    x_feats = np.reshape(x_feats,(1,x_feats.shape[0]))
    x_pca_feat = sc.transform(x_feats) 
    x_pca_feat = pca.transform(x_pca_feat)
    x_pca_feat = x_pca_feat[:,:350]
    pdb.set_trace()
    return {'mel':logmel, 'pca':x_pca_feat}

def create_model():
    model = create_mel_and_pca_model()
    #model.load_weights(model_weights)
    return model

def generate_tag(mel_spec,model):
    prediction = np.log(np.ones((1,41)))
    #mel_spec = np.reshape(mel_spec,(1,mel_spec.shape[0],mel_spec.shape[1],1))
    for i in range(10):
        model.load_weights('model_outs/mel_and_pca_model/fold{}/best_model_3.h5'.format(i))
        for req_mel_len in [263,363,463,563,663,763]:
            for _ in range(5):
                this_pred = model.predict(mel_spec)
                prediction = prediction + np.log(this_pred + 1e-38)
                del this_pred
    #geometric average of all the predictions
    prediction = prediction/300.
    prediction = np.exp(prediction)
    return prediction


def get_tags(filepath):
    data = get_mel_and_pca(filepath)
    model = create_model()
    prediction = generate_tag(data,model)
    labels= pd.read_csv('train.csv').label.tolist()
    labels = list(sorted(list(set(labels))))
    top_3 = np.array(labels)[np.argsort(-prediction, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    return predicted_labels
    # print(labels[np.argmax(prediction)])
#def main(args):
 #   weights = args.weights
  #  model = create_model(weights)
   # file = args.file
    ##import pdb
    #pdb.set_trace()
    #filepath = file[0]

    #data  = get_mel_and_pca(filepath)
    #prediction = generate_tag(data,model)
    #labels = pd.read_csv('/hdd0/datasets/MuseTek/Data/freesound-data/train.csv').label.tolist()
    #labels = list(sorted(list(set(labels))))
    #top_3 = np.array(labels)[np.argsort(-prediction, axis=1)[:, :3]]
    ##predicted_labels = [' '.join(list(x)) for x in top_3]
    #print(predicted_labels)
    # print(labels[np.argmax(prediction)])
    #print("hello")
    #print(prediction)

if __name__=='__main__':
    #import argparse
    #parser = argparse.ArgumentParser(description="predicts which class file(s) belong(s) to")
    #parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
    #    help='weights file in hdf5 format', default="weights.hdf5")
    #parser.add_argument('file', help="file(s) to classify", nargs='+')
    #args = parser.parse_args()
    #print("hello")
    #main(args)
    tags = get_tags('/hdd0/datasets/MuseTek/Data/Landr copy/Adrien Fertier_Lo-fi Rock/synth_note_piano_C2.wav.mp3')


    





