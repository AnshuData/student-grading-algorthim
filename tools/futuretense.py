from fuzzywuzzy.fuzz import ratio
from fuzzywuzzy import fuzz
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag


def futuretense(sentence):

    future_tense_modals = ["may", "might", "should", "would", "will","shall","gonna"]
    """
    Check if a sentence is in future tense
    """
    # Tokenize the sentence and perform POS tagging

    tokens = sentence.split()
    tagged_tokens = nltk.pos_tag(tokens)
    tags = [pos_tags[1] for pos_tags in tagged_tokens]
    
    pattern = "what will happen next"
    score = fuzz.token_set_ratio(sentence, pattern)
    if score >= 70:
        return True
    

    for token, tag in tagged_tokens:
        # Check if the main verb is in the future tense
        if token.lower() in future_tense_modals:
            return True
            
            
        elif "be" in tokens and "going" in tokens and "to" in tokens:
            return True
        
        
        elif "will" in tokens and "be" in tokens and "ing" in token:
            return True
        
        else :
            for k,i in enumerate(tags):
                try:
                    if (i == "VBG") and (tags[k-1] =="VBZ" or tags[k-1] =="VBP") and tags[k+1] == 'TO' :
                        return(True)
                        
                except IndexError:
                    pass
                
    
    return False

                
    