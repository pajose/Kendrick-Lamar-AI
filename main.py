#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 23:07:49 2018

@author: CTPoonage
"""

from LSTMModel import LSTMModel
from BaseMarkovModel import BaseMarkovModel

if __name__ == '__main__':
    
    artist = "kendrick_lamar"
    file_path = str(artist) + "_lyrics.txt"
    taggedTokens = "taggedTokens.pkl"
    batch_size = 2
    epochs = 30
    max_syllables = 14
    
    # Base Markov Model
    print("~~~~~~~~~~~~~~BASE MARKOV MODEL~~~~~~~~~~~~~~")
    BaseModel = BaseMarkovModel(file_path,taggedTokens)
    final_rap = ""
    for i in range(3):
        verse = BaseModel.createVerse(9, 6)
        final_rap += verse + "\n"
        print(verse)
    with open(str(artist) + "_base_markov_rap.txt", 'w', encoding='utf8') as f:
        [f.write(final_rap)]
    print("\nFinished writing to " + str(artist) + "_base_markov_rap.txt!")
    
    # LSTM Model
    print("~~~~~~~~~~~~~~LSTM MODEL~~~~~~~~~~~~~~")
    neural_network_model = LSTMModel(artist, file_path, batch_size, epochs, max_syllables, training=False)
    
    num_lines = 24 # Number of lines after the seed line (number of lines to generate = num_lines + 1)
    
    if neural_network_model.training:
        lyrics = neural_network_model.get_artist_lyrics(file_path)
    else:
        lyrics = neural_network_model.generate_lyrics()
        
    rhyming_endings = neural_network_model.get_rhyming_endings(lyrics)
    
    rap_analytics = neural_network_model.get_lyric_analytics(lyrics, rhyming_endings)
    
    if neural_network_model.training:
        X, y = neural_network_model.build_dataset(rap_analytics)
        neural_network_model.train(X, y)
    else:
        vector_data = neural_network_model.vectorize_rap(file_path, rhyming_endings, num_lines)
        final_rap = neural_network_model.convert_vector_to_rap(vector_data, rap_analytics, lyrics, rhyming_endings)
        with open(str(artist) + "_neural_network_rap.txt", 'w', encoding='utf8') as f:
            [f.write(lyric+"\n") for lyric in final_rap]
        print("\nFinished writing to " + str(artist) + "_neural_network_rap.txt!")
        