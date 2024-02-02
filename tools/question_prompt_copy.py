
from nltk import sent_tokenize, word_tokenize, pos_tag
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')

#from nltk.tokenize import word_tokenize



def check_prompt_copying(prompt, stu_response):
    """ checks if student response is copied from prompt """

    copied = False  # Define copied as False initially
    
    try :
        prompt_sentences = sent_tokenize(prompt)
        student_sentences = sent_tokenize(stu_response)


        for sentences in prompt_sentences:

            for student_responses in student_sentences:

                prompt_vector = model.encode(sentences)
                student_ans_vector = model.encode(student_responses)
                text_similarity = cosine_similarity([prompt_vector], [student_ans_vector])

                if text_similarity >= 0.95:
                    copied = True  # Update copied to True if similarity threshold is met

                    break
                    
    except (TypeError, AttributeError) :
            pass


        
    return copied


def check_question_copying(question, stu_response):
    question_copied = False  # Define copied as False initially
    
    question_tokenized = sent_tokenize(question)
    student_sentences = sent_tokenize(stu_response)
    
    for sentences in question_tokenized:
        
        for student_responses in student_sentences:
        
            question_vector = model.encode(sentences)
            student_ans_vector = model.encode(student_responses)
            text_similarity = cosine_similarity([question_vector], [student_ans_vector])
            
            if text_similarity >= 0.76:
                answer_copied = True  # Update copied to True if similarity threshold is met
                #print(text_similarity,student_responses,sentences)
                break
                
                
    return question_copied
                