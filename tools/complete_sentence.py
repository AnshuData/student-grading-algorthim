
import nltk

def is_complete_sentence(sentence):
    """
    Returns True if the given sentence is a complete sentence,
    False otherwise. Based on the presence of subject nd predicate
    """
    # Tokenize the sentence into words
    words = sentence.split()
    
    # Check if sentence has a subject and a predicate
    if any([t[1].startswith('N') for t in nltk.pos_tag(words)]) \
            and any([t[1].startswith('V') for t in nltk.pos_tag(words)]):
        return True
    else:
        return False