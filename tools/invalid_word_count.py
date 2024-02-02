
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize, word_tokenize, pos_tag

import nltk
import enchant

enchant_dict = enchant.Dict("en_US")

def invalid_words(sentence):

    """ counts the number of invalid words based on their presence in the enchant dictionary """

    if sentence is None:
        return None
    
    # Tokenize the sentence into individual words
    words = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)

    # Check the spelling of each word
    invalid_words = []
    
    for word in pos_tags : 
        
        if enchant_dict.check(word[0]) == False and word[1] != "NNP":
            invalid_words.append(word)
        
    return len(invalid_words)
