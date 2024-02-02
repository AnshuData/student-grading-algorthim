
import spacy.cli
spacy.cli.download("en")
import re
from gramformer import Gramformer
gf = Gramformer(models = 1, use_gpu=False)

#if using gingerit
#from gingerit.gingerit import GingerIt
#gingerit = GingerIt()



def grammar_check_gramformer(answer):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', answer)
    corrected_sentences = []
    corrections = []
    
    for sentence in sentences:
        if len(sentence.split()) <= 52:
            # Correct the grammar using Gramformer library
            gramformer_corrected = list(gf.correct(sentence, max_candidates=1))[0]

            #if want to use gingerit as well

            #ginger_it_corrected = str(gingerit.parse(gramformer_corrected)['result'])
            #original_vs_corrected = gf.get_edits(sentence, ginger_it_corrected)
            #corrections = [corrections[1::3]for corrections in original_vs_corrected]
        
            # Get the edits made by Gramformer and GingerIt libraries
            original_vs_corrected = gf.get_edits(sentence, gramformer_corrected)
        
            # Extract the corrections made by GingerIt library
            correction = [correction[1::3] for correction in original_vs_corrected]
        
            # Append the corrected sentence and the corrections to their respective lists
            corrected_sentences.append(gramformer_corrected)
            corrections.append(correction)

        else:
            # Split the sentence into chunks of 52 words since gramformer cannot handle sentences made up of words > 50
            chunks = [sentence.split()[i:i+52] for i in range(0, len(sentence.split()), 52)]
            chunk_sentences = []
            chunk_corrections = []
            
            for chunk in chunks:
                # Combine the chunk into a sentence
                chunk_sentence = ' '.join(chunk)
                
                # Correct the grammar using Gramformer library
                gramformer_corrected = list(gf.correct(chunk_sentence, max_candidates=1))[0]
        
                # Get the edits made by Gramformer and GingerIt libraries
                original_vs_corrected = gf.get_edits(chunk_sentence, gramformer_corrected)
        
                # Extract the corrections made by GingerIt library
                correction = [correction[1::3] for correction in original_vs_corrected]
                
                # Append the corrected chunk sentence and the corrections to their respective lists
                chunk_sentences.append(gramformer_corrected)
                chunk_corrections.append(correction)
            
            # Append the corrected chunk sentences and the corrections to their respective lists
            corrected_sentences.append(' '.join(chunk_sentences))
            corrections.append([my_tuple for inner_list in chunk_corrections for my_tuple in inner_list])
    
    correction_list = [my_tuple for inner_list in corrections for my_tuple in inner_list]
    return ' '.join(corrected_sentences), correction_list
