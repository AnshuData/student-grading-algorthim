
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def idk(idk_list, student_response) :

    """ check for the presence of sentences like idk, i don't know in the student response """
    """ using cosine similarity """

    idk_present = False
    
    try :
        
        sentences = student_response.split(".")

        for text in idk_list:
            
            for sentence in sentences:
                tokens1 = [w.lower() for w in word_tokenize(text) if w.lower() not in stop_words]
                tokens2 = [w.lower() for w in word_tokenize(sentence) if w.lower() not in stop_words]
                
                intersection = len(set(tokens1).intersection(set(tokens2)))
                union = len(set(tokens1).union(set(tokens2)))
                jaccard_similarity = intersection / union
                
                if jaccard_similarity >= 0.2:
                    #print(text, sentence, text_similarity)
                    idk_present = True
                    break
                    
    except (TypeError, AttributeError):
        pass
    
    
    return idk_present

    
  
