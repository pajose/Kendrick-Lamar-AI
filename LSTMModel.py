#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 20:17:14 2018

@author: CTPoonage
"""

import markovify
import operator
import os
import pronouncing
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop

class LSTMModel:
    def __init__(self, artist, file_path, batch_size, epochs, max_syllables, training=False):
        '''
            Initializes an LSTM model to train and generate lyrics.
            
            @param artist: The name of the artist. This is used to make customized txt files
            @param file_path: A string indicating the name of the lyric file to open
            @param batch_size: The amount of training examples to train at once
            @param epochs: The number of full training cycles on the dataset
            @param max_syllables: The maximum number of syllables each line must have
            @param training: A boolean flag indicating whether or not to train the model; default is False
        '''
        self.artist = artist
        self.file_path = file_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_syllables = max_syllables
        self.training = training
        self._model = self.build_model()
        self._markov = self.markov(file_path)

    def build_model(self):
        '''
            Builds and returns a Keras Sequential model.
            
            @return: The Sequential model used to train against the lyric data.
        '''
        model = Sequential()
        model.add(LSTM(32, input_shape=(1,2), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(Dense(16))
        model.add(LSTM(2, return_sequences=True))
        
        model.summary()
        
        optimizer = RMSprop(lr=0.01)
        model.compile(loss='mse', optimizer=optimizer)
    
        if str(self.artist)+".h5" in os.listdir(".") and self.training == False:
            print("Loading saved weights from " + str(self.artist) + ".h5\n")
            model.load_weights(str(self.artist) + ".h5")
        
        return model
    
    
    def train(self, input_data, output_data):
        '''
            Train and fit a model based on the input and output data from the original lyrics.
            Given an input and output, the model will be able to train itself on patterns between one line and the next line.
            
            @param model: The Keras model we will be using to train
            @param input_data: A list of "first" lines with rhyme and syllable data
            @param output_data: A list of lines following the "first" lines with rhyme and syllable data
        '''
        self._model.fit(input_data, output_data, batch_size=self.batch_size, epochs=self.epochs)
        
        # Save the weights to load later and reduce computation
        self._model.save_weights(str(self.artist)+".h5")
    
    
    def markov(self, path):
        '''
            A helper function to create a Markov model given a file path.
            
            @param path: A file path
            @return: A markov model which we can use to generate random sentences.
        '''
        with open(path, 'r', encoding='utf8') as f:
            markov_model = markovify.NewlineText(f)
        return markov_model
    
    
    def syllables(self, line):
        '''
            The total number of syllables in a given line, divided by max_syllables in order
            for the neural network to work with a normalized value between 0 and 1. Lines with a
            total / max_syllable ratio greater than 1 are discarded.
            
            @param line: A line from the lyrics
            @return: The total / max_syllable ratio.
        '''
        total = 0
        vowels = 'aeiouy'
        words = line.split()
        
        # If there are no words, then there are no syllables
        if len(words[0]) == 0:
            return 0
        # If there are words, then we calculate syllables by checking against the vowels, including some special cases
        else:
            for word in words:
                count = 0
                if word[0] in vowels:
                    count += 1
                for i in range(1, len(word)):
                    if word[i] in vowels and word[i-1] not in vowels[:-1]:
                        count += 1
                if word.endswith('e'):
                    count -= 1
                if word.endswith('le') and word[-3] not in vowels:
                    count += 1
                if count == 0:
                    count += 1
                total += count
                    
        return total / self.max_syllables
    
    
    def get_artist_lyrics(self, path):
        '''
            Puts all of the artist's lyrics into a list, which the model will use for predictions.
            
            @param path: A file path containing the artist's lyrics
            @return: A list of the artist's lyrics
        '''
        with open(path, 'r', encoding='utf8') as f:
            artist_lyrics = f.read().split("\n")
            
            # Remove empty strings from the lyrics
            artist_lyrics = list(filter(None, artist_lyrics))
            
            # Remove ellipses from the lyrics
            while "..." in artist_lyrics:
                artist_lyrics.remove("...")
        return artist_lyrics
    
    
    def generate_lyrics(self):
        '''
            Build a list of lyrics, which are random sentences from the Markov model.
            Do not add a lyric if the last word has been used more than 5 times.
        
            @return: The list of lyrics to generated by the Markov model.
        '''
        lyrics = [] # a list of the lyrics we want to generate
        last_words = [] # used to make last words more unique in the lyrics
        
        with open(self.file_path, 'r', encoding='utf8') as f:
            num_lines_in_file = len(f.read().split("\n"))
        
        # Generate num_lines_in_file / 5 possible lyrics
        while len(lyrics) < num_lines_in_file / 5:
            line = self._markov.make_sentence()
            
            # There is a chance that the Markov model might fail to make a sentence, so we need some type checking
            if type(line) != type(None) and self.syllables(line) < 1:
                # Strip away any punctuation from the last word.
                last_word = line.split()[-1].strip(" -',.:&")
                last_words.append(last_word)
            
                if line not in lyrics and last_words.count(last_word) < 5:
                    lyrics.append(line)
        
        return lyrics
    
    def rhyme_ending(self, line):
        '''
            Helper function to get the rhyme ending of one line. A rhyme ending is defined as the last syllable of the last word.
            
            @param line: The line whose rhyme ending we want to get.
            @return: The most common rhyme ending associated with the line
        '''
        # Strip away any punctuation from the last word.
        last_word = line.strip(" -',.:&").split()[-1]
        
        # Get the rhyme of the last word with Pronouncing
        rhymes = pronouncing.rhymes(last_word)
    
        rhyme_endings = []
        vowels = 'aeiouy'
        for rhyme in rhymes:
            # If the last character of the rhyme is a vowel, then simply add the last two characters
            # Otherwise, iterate backwards from the end of the rhyming word until we find a vowel.
            if rhyme[-1] in vowels:
                rhyme_endings.append(rhyme[-2:])
            for i in range(len(rhyme) - 2, -1, -1):
                if rhyme[i] in vowels:
                    if rhyme[i-1] in vowels:
                        rhyme_endings.append(rhyme[i-1:])
                    else:
                        rhyme_endings.append(rhyme[i:])
                    break
                
        # If the last word does not rhyme with anything, then we just return the last two characters of the word.
        # Otherwise, we get the most common rhyme ending from the list of rhymes.
        if len(rhyme_endings) == 0:
            return last_word[-2:]
        else:
            return max(set(rhyme_endings), key=rhyme_endings.count)    
    
    def get_rhyming_endings(self, lyrics):
        '''
            Writes all the rhyme endings to a file which can be loaded in the future.
            
            @param lyrics: a list of all the generated lyrics.
            @return: a list of all rhyming endings in from the lyrics
        '''
        # Load the rhymes file for faster computation if we're not training the model
        if str(self.artist) + "_rhymes.txt" in os.listdir(".") and self.training == False:
            print("Loading saved rhyme endings from " + str(self.artist) + "_rhymes.txt\n")
            with open(str(self.artist) + "_rhymes.txt", "r", encoding="utf8") as f:
                final_rhyming_endings = f.read().split("\n")
            return final_rhyming_endings
        
        # If we're training the model, find all the rhyme endings within the original lyrics.
        else:        
            all_rhyming_endings = []
            
            for line in lyrics:
                all_rhyming_endings.append(self.rhyme_ending(line))
            
            all_rhyming_endings = list(set(all_rhyming_endings))
            
    #        We need to reverse the rhymes so that the ending sounds are closer together.
    #        This will give the neural network less chance to mess up a rhyme when picking from the list,
    #        considering it will likely pick the neighbor of the desired rhyme.
    #        
    #        Example:
    #            Let's say the neural network is looking for a rhyme that ends in -y.
    #            These would be the sorted non-reversed and reversed lists
    #            
    #            Non-reversed list (The -y rhyme endings are ly and ny, but many other rhyme endings separate them):
    #                ly, ma, me, mm, mp, ms, nd, ne, ng, nk, ns, nt, ny
    #            Reversed list (all these rhymes end with -y, making a neural network rhyme mistake less severe):
    #                ay, cy, dy, ey, gy, ly, my, oy, ry, ty
    #            
    #            Rhymes rely on the ending sound, so reversing the list
    #            will reduce error and ensure we get the desired rhyme "sound."
                    
            reverse_rhymes = [rhyme[::-1] for rhyme in all_rhyming_endings]
            reverse_rhymes = sorted(reverse_rhymes)
            
            final_rhyming_endings = [rhyme[::-1] for rhyme in reverse_rhymes]
            
            # Write all the rhymes to a rhymes file which can be loaded for faster computation
            with open(str(self.artist) + "_rhymes.txt", "w", encoding="utf8") as f:
                f.write("\n".join(final_rhyming_endings))
            
            return final_rhyming_endings
    
    def rhymes(self, line, rhyming_endings):
        '''
            Returns the line's rhyme as a float between 0 and 1 to feed to the neural network.
            
            @param line: The line whose rhyme ending we want to get.
            @param rhyming_endings: A list of rhyme endings we use to grab the index of the rhyme in the line.
        '''
        rhyme = self.rhyme_ending(line)
        # There is a chance the last word that the Markov model generates does not rhyme with anything.
        # So we must use try/except in order to handle this behavior.
        # A line with no rhyme simply gets a value of 0.
        
        try:
            rhyme_index = rhyming_endings.index(rhyme)
            return float(rhyme_index / len(rhyming_endings))
        except ValueError:
            return 0
    
    
    def get_lyric_analytics(self, lyrics, rhyming_endings):
        '''
            A helper function to get the rhyme and syllable data for each line in a list of lyrics.
            
            @param lyrics: A list of lyrics
            @param rhyming_endings: A list of all the rhyming endings
            
            @return: A 2D list containing. Each inner list contains three values: [line, rhyme_data, syllable_data].
        '''
        analytics = []
        for line in lyrics:
            analytics.append([line, self.rhymes(line, rhyming_endings), self.syllables(line)])
        return analytics
        
    
    def build_dataset(self, analytics):
        '''
            Builds a dataset from the rap analytics based on the rhyme and syllable data.
            Iterates through all the analytics and adds to the input (x) data and output (y) data based on the following:
                x = [rhyme_0, syllable_0]
                y = [rhyme_1, syllable_1]
            
            @param analytics: A list of rap analytics, which contains a line and the line's rhyme and syllable data for each entry.
            
            @return: The input (x) and output (y) data for the model to train on.
        '''
        input_data = []
        output_data = []
        
        for i in range(len(analytics) - 1):
            line1_info = analytics[i][1:]
            line2_info = analytics[i+1][1:]
            
            x = [line1_info[0], line1_info[1]]
            x = np.array(x)
            x = x.reshape(1,2)
            input_data.append(x)
            
            y = [line2_info[0], line2_info[1]]
            y = np.array(y)
            y = y.reshape(1,2)
            output_data.append(y)
        
        input_data = np.array(output_data)
        output_data = np.array(output_data)
        
        #print(input_data.shape)
        #print(output_data.shape)
        
        return input_data, output_data
    
    
    def vectorize_rap(self, path, rhyming_endings, num_lines):
        '''
            The final function that translates lyrics into numbers, which the neural network will understand and use to predict.    
        
            @param model: The neural network model that we will use to predict lyrics for the rap
            @param path: A file path containing the artist's real lyrics
            @param rhyme_endings: A list created from get_rhyming_endings that contains all rhyme_endings from a given artist
            @return: Vectors that will be used to generate raps
        '''
        vectorized_rap_data = []
        artist_lyrics = self.get_artist_lyrics(path)
        
        first_line_index = np.random.choice(len(artist_lyrics) - 1)
        first_lines = artist_lyrics[first_line_index:first_line_index+10]
        
        starting_lines = []
        
        for line in first_lines:
            starting_lines.append([self.rhymes(line, rhyming_endings), self.syllables(line)])
        
        starting_vector = self._model.predict(np.array([starting_lines]).flatten().reshape(10, 1, 2))
        vectorized_rap_data.append(starting_vector)
        
        for i in range(num_lines): # Number inputted into range will be the number of lines to generate
            current_vector = self._model.predict(np.array([vectorized_rap_data[-1]]).flatten().reshape(10, 1, 2))
            vectorized_rap_data.append(current_vector)
        
        return vectorized_rap_data
    
    
    def convert_vector_to_rap(self, vectors, analytics, generated_lyrics, rhyming_endings):
        '''
            Converts vector representations of lyrics into the actual lyrics in word form.
            
            @param vectors: Vector representations of the rap that were predicted by the neural network
            @param analytics: A list containing the rhyme and syllable data for each line
            @param generated_lyrics: The generated lyrics from the Markov model
            @param rhyming_endings: A list created from get_rhyming_endings that contains all rhyme_endings from a given artist
        '''
        
        def calculate_uncreativity(rap, line2):
            '''
                A helper function to penalize the neural network if it uses a line with an ending word that has already been used.
                
                @param rap: A list of lines that were chosen to be in the generated rap
                @param line2: One line that is evaluated against all the lines within the rap list.
                @return: A float that represents uncreativity. The higher the number, the lower the lyric score.
            '''
            uncreativity = 0
            for line1 in rap:
                last_word1 = line1.split()[-1].strip(" -',.:&")
                last_word2 = line2.split()[-1].strip(" -',.:&")
                if last_word1 == last_word2:
                    uncreativity += 0.3
            return uncreativity
        
        def calculate_lyric_score(vector, rhyme, syllables, uncreativity):
            '''
                Calculates a score based on differences between the original lyric and generated lyric in order to find the lyric with the best fit to the model.
                We must calculate the predicted_rhyme and predicted_syllables in order to get the correct rhyme endings and number of syllables for our generated rap.
                The highest possible score, though unlikely, is 1. There is no limit on how low a score can be. 
                
                @param vector: A list of rhyme/syllable data predictions within the vectorized rap data
                @param rhyme: The rhyme value of the generated lyric
                @param syllables: The syllable value of the generated lyric
                @param uncreativity: The amount of uncreativity that the generated lyric adds to the rap
                @return: The lyric score. The highest score gets chosen to be part of the final generated rap
            '''
            predicted_rhyme = vector[0] * len(rhyming_endings)
            predicted_syllables = vector[1] * self.max_syllables
            
            score = 1 - (abs(predicted_rhyme - float(rhyme)) + abs(predicted_syllables - float(syllables))) - uncreativity
            
            return score
        
        rap = []
        
        vector_list = []
        
        for vector in vectors:
            vector_list.append(list(vector[0][0]))
        
        for vector in vector_list:
            line_scores = []
            all_lines = []
            for data in analytics:
                line, rhyme, syllables = data[0], data[1], data[2]
                uncreativity = calculate_uncreativity(rap, line)
                all_lines.append(line)
                    
                score = calculate_lyric_score(vector, rhyme, syllables, uncreativity)
                line_scores.append([line, score])
            
            line_scores.sort(key=operator.itemgetter(1), reverse=True)
            
            line, score = line_scores[0][0], line_scores[0][1]
            cleaned_line = line.strip(" -',.:&").capitalize()
            print(cleaned_line) #print the line to the console if desired
            rap.append(cleaned_line)
    
            best_line_index = all_lines.index(line)
            all_lines.pop(best_line_index)
    
        return rap