import string


def remove_punctuations_and_spaces(sentence):
    # Remove punctuations
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    
    # Remove spaces
    sentence = sentence.replace(' ', '')
    
    return sentence


def filter_spell_errors(tuples_list):
    # Create a list to hold the filtered tuples
    filtered_list = []
    
    # Create a set of all punctuation characters
    punc_set = set(string.punctuation)
    
    # Loop through each tuple in the list
    for tup in tuples_list:
        # Get the two words from the tuple
        word1, word2 = tup
        
        # Remove any punctuation from the second word
        word2 = ''.join(ch for ch in word2 if ch not in punc_set)
        
        # Check if the two words are equal after removing any punctuation and capitalization
        if word1.lower() == word2.lower() or remove_punctuations_and_spaces(word1.lower()) == remove_punctuations_and_spaces(word2.lower()):
            continue
        
        # Check if the second word is a spelling correction of the first word
        if word1.lower() in word2.lower() or word2.lower() in word1.lower():
            continue
        
        # If the two words are different and the second word is not a spelling correction of the first word, add the tuple to the filtered list
        filtered_list.append(tup)
   # print(filtered_list)
    return filtered_list


