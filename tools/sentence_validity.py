
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag

import enchant
enchant_dict = enchant.Dict("en_US")
import spacy
#nlp = spacy.load('en_core_web_lg')
nlp = spacy.load('en_core_web_sm')



def check_validity(sentence):
    if sentence is None:
        return None
    
    # Tokenize the sentence into individual words
    words = word_tokenize(sentence)

    # Check the spelling of each word
    valid_words = []
    invalid_words = []

    for word in words:
        if enchant_dict.check(word):
            valid_words.append(word)

        else:
            invalid_words.append(word)

    if len(invalid_words) >= 3:
        return "invalid"
    else:
        return None


def sensible(sentence):
    # Parse the sentence using spaCy
    doc = nlp(sentence)

    # Compute the sum of pairwise cosine similarities between words
    score = 0.0
    count = 0
    for i in range(len(doc)):
        for j in range(i+1, len(doc)):
            score += doc[i].similarity(doc[j])
            count += 1

    # Compute the average cosine similarity
    if count > 0:
        score /= count
        
    if score < 0.35:
        return False
    else:
        return True

