import re

def count_sentence(student_response):

    """ counts the number of sentences in a student response based on the punctuations """

    # Split the text into sentences using regular expression
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)(?=\s?\w)', student_response)

    
    
    # Return the number of sentences

    return len(sentences)
