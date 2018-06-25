# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:49:39 2018

@author: patri
"""
import pronouncing
import nltk, re
from nltk.corpus import brown
import random
from nltk.collocations import *
from pickle import dump
from pickle import load

class BaseMarkovModel:
    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    PPronouns = ['PRP$', 'PP$', 'WP', 'WP$']
    
    def __init__(self, rawLyrics, taggedTokens=None):
        corpus = open(rawLyrics, encoding='utf8')
        self._rawLyrics = corpus.read()
        self._tokens = self.tokenizeLyrics(self._rawLyrics)
        self._transitionProbabilities = self.createMarkovChain(self._tokens)
        if (taggedTokens == None):
            self._taggedTokens = self.createTaggedTokens(self._tokens)
        elif(type(taggedTokens) == str and taggedTokens.endswith('.pkl')):
            self._taggedTokens = self.loadTaggedTokens(taggedTokens)
            
    def tokenizeLyrics(self, rawLyrics):
        
        noNewLine = re.sub("\n", " ", rawLyrics)
        tokens = noNewLine.split(' ')
        tokens = list(filter(None, tokens))
        
        return tokens
    
    
    def createMarkovChain(self, tokens):
        
        freqDict = {}
        # For each word in the vocabulary, create a count of how many times the next word appears next to it
        for i in range(len(tokens) - 1):
            currWord = tokens[i]
            nextWord = tokens[i+1]
    
            if (currWord not in freqDict):
                freqDict[currWord] = {nextWord: 1}
            else:
                if (nextWord not in freqDict[currWord]):
                    freqDict[currWord][nextWord] = 1
                else:
                    freqDict[currWord][nextWord] += 1
                
        transitionProbabilities = {}
        
        # Convert counts to probabilities
        for currWord in freqDict:
            total = sum(freqDict[currWord].values())
            transitionProbabilities[currWord] = {}
            
            for nextWord in freqDict[currWord]:
                transitionProbabilities[currWord][nextWord] = freqDict[currWord][nextWord] / total
                
        
        return transitionProbabilities
    
    
    def POStagging(self, tokens):
        
        # Automatically tag each word
        taggedTokens = dict(nltk.pos_tag(tokens))
        
        # A word can have multiple parts of speech, so store the POS that is most frequently used
        fd = nltk.FreqDist(brown.words(categories='news'))
        cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
        most_freq_words = fd.most_common(500)
        # If a word from the 500 most frequent words in our corpus is in our tokens, use that tag (because its more accurate)
        for (word, _) in most_freq_words:
            if(word in taggedTokens):
                taggedTokens[word] = cfd[word].max()
                
        taggedTokens['i'] = 'PRP'
        taggedTokens['i\m'] = 'VBP'
    
        return taggedTokens
    
    def createTaggedTokens(self, tokens):
        ''' Creates a dictionary of tagged tokens and saves it into a pkl file'''
        taggedTokens = self.POStagging(tokens)
        output = open('taggedTokens.pkl', 'wb')
        dump(taggedTokens, output, -1)
        output.close()
        
        return taggedTokens
    
    def loadTaggedTokens(self, taggedTokensFile):
        print('Loading saved tagged tokens from', taggedTokensFile, "\n")
        taggedInput = open(taggedTokensFile, 'rb')
        taggedTokens = load(taggedInput)
        taggedInput.close()
        
        return taggedTokens

    
    def chooseNextWord(self, currWord, transitionProbabilities, taggedTokens):
        
        randProb = random.random()
        currTotalProbability = 0.0
        
        # If currWord is a pronoun or the word 'at', create a new markov chain that doesn't contain verbs
        if(taggedTokens[currWord] in self.PPronouns or taggedTokens[currWord] in ['AT', 'PPSS']):
            newMarkovChain = dict(item for item in transitionProbabilities[currWord].items() if taggedTokens[item[0]] not in self.verbs)
            # Normalize new markov chain for a specific word and condition
            total = sum(newMarkovChain.values())
            for word in newMarkovChain:
                newMarkovChain[word] = newMarkovChain[word] / total
                
            # Choose new word
            for nextWord in newMarkovChain:
                currTotalProbability += newMarkovChain[nextWord]
                
                if (randProb <= currTotalProbability):
                    return nextWord
        
        else:
            # Choose next word based on Markov Chain transition states
            for nextWord in transitionProbabilities[currWord]:
                currTotalProbability += transitionProbabilities[currWord][nextWord]
                
                if (randProb <= currTotalProbability):
                    return nextWord
    
    
    def createVerse(self, lengthB, lengthV):
        
        verse = ""
        lastWord = ""
        for lineNum in range(lengthV):
        # Create a line with length lengthB
            # Choose a random word to start line
            newWord = random.choice(list(self._transitionProbabilities.keys()))
            line = newWord.capitalize() + " "
    #       Choose next lengthB words for the line (minus 1 because we already created first word)
            for word in range(lengthB - 1):
                newWord = self.chooseNextWord(newWord, self._transitionProbabilities, self._taggedTokens)
                line += newWord + " "
                # Store the last words for odd lines to find a rhyme
                if(word == lengthB - 2): # and (lineNum + 1) % 2 == 1):
                # If last word of a line is a coordinating conjunction, add another word
                    if(self._taggedTokens[newWord] in ['CC', 'AT', 'RB', 'PPSS', 'WRB', 'TO', 'IN', 'PPS', 'PP$'] or newWord == 'i'):
                        newWord = self.chooseNextWord(newWord, self._transitionProbabilities, self._taggedTokens)
                        line += newWord + " "
                    if((lineNum + 1) % 2 == 1):
                        lastWord = newWord
    
            if((lineNum + 1) % 2 == 0):
                if(re.match(r'[()"\'{}*;,]', lastWord[-1])):
                    lastWord = lastWord[0:-1]
                possibleRhymes = pronouncing.rhymes(lastWord)
                if(possibleRhymes):
                    rhymesPOS = dict(nltk.pos_tag(possibleRhymes))
                    possibleRhymes = (word for (word,_) in rhymesPOS if rhymesPOS[word] not in ['CC', 'AT', 'RB', 'PPSS', 'WRB', 'TO', 'IN', 'PPS', 'PP$'] or newWord == 'i')
                    rhyme = random.choice(pronouncing.rhymes(lastWord))
                    line += rhyme
            verse += line + "\n"
            
        return verse
            
    
if __name__ == '__main__':

    
    KendrickAI = BaseMarkovModel('kendrick_lamar_lyrics.txt')
    
#    vocabulary = set(tokens)
#    lengthVocab = len(vocabulary)
#    

    # Used for debugging
#    tagToLookFor = 'PRP'
#    for (word,tag) in taggedTokens.items():
#        if(tag == tagToLookFor):
#            print(word, 'is a', tagToLookFor)

    for i in range(3):
        verse = KendrickAI.createVerse(7, 6)
        print(verse)
