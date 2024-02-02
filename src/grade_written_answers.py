import datetime
import json

import nltk
import spacy.cli
import uvicorn
from fastapi import Body, FastAPI
from gibberish_detector import detector
from gramformer import Gramformer
#from gingerit.gingerit import GingerIt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from tools.complete_sentence import is_complete_sentence
from tools.compound_sentence import conjuction
from tools.count_sentences import count_sentence
from tools.filter_spell_errors import filter_spell_errors
from tools.futuretense import futuretense
from tools.grammar_check_gramformer import grammar_check_gramformer
from tools.idk import idk
from tools.invalid_word_count import invalid_words
from tools.keyword_presence import kw_presence
from tools.keyword_presence_Q5 import kw_presence_Q5
from tools.keyword_presence_Q6 import kw_presence_Q6
from tools.question_prompt_copy import (check_prompt_copying,
                                        check_question_copying)
from tools.sentence_validity import check_validity, sensible
from tools.keyword_presence_spoken import kw_presence_spoken
from tools.opinion import opinion_justified

spacy.cli.download("en")


def wtype1(input_file):
    """
    Perform autograding for a specific writing type (Type 1).

    Args:
    - input_file (dict): Input file containing student response, ideal answer, prompt, and question.

    Returns:
    - str: JSON representation of the autograding results.
    """

    # extract student_response, ideal answer, prompt and question from input
    student_response = input_file["transcript"]
    ideal_answer = input_file["metadata"]["ideal_answer"]
    prompt = input_file["metadata"]["prompt"]
    question = input_file["metadata"]["question"]


    input_file["data"] = {}

    # Check for question copying
    copied_from_question = check_question_copying(question, student_response)

    if student_response is None or len(student_response.split()) <= 2 or Detector.is_gibberish(student_response.lower()) is True\
        or student_response.lower() == prompt.lower() or check_question_copying == True:

        # Handle cases where response is invalid or copied
        input_file["score"] = 0
        input_file["data"]["grammar_check_output"] = ""
        input_file["data"]["grammatical_mistakes"] = 0
        input_file["data"]["corrections_made"] = None
        input_file["data"]["similarity_to_ideal_answer"] = 0


    else:
        # Perform grammar check using Gramformer
        grammar_check = grammar_check_gramformer(student_response)
        gram_corrected = grammar_check[0]
        corrections = grammar_check[1]


        # Calculate cosine similarity between student response and ideal answer
        ideal_answervector = model.encode(ideal_answer)
        student_ans_vector = model.encode(student_response)
        similarity = cosine_similarity([ideal_answervector], [student_ans_vector])


        input_file["data"]["grammar_check_output"] = gram_corrected 
        input_file["data"]["grammatical_mistakes"] = len(corrections)
        input_file["data"]["corrections_made"] = corrections
        input_file["data"]["similarity_to_ideal_answer"] =  round(float(similarity[0][0]),1)

        # Determine the score based on similarity and grammatical mistakes
        if input_file["data"]["similarity_to_ideal_answer"] <= 0.5 or input_file["data"]["grammatical_mistakes"] >= 5:
            input_file["score"] = 0

        elif 0.5 < input_file["data"]["similarity_to_ideal_answer"] < 0.8 or 2 <input_file["data"]["grammatical_mistakes"] < 5:
            input_file["score"] = 1
    
        elif input_file["data"]["similarity_to_ideal_answer"] >= 0.8 and input_file["data"]["grammatical_mistakes"] <= 2:
            input_file["score"] = 2

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output

import json
import datetime
from langid.langid import LanguageIdentifier, model
from gingerit.gingerit import GingerIt
from pywebcopy import save_webpage

identifier = LanguageIdentifier.from_modelstring(model)
gf = GingerIt()

def wtype2(input_file):
    """
    Perform autograding for a specific writing type (Type 2).

    Args:
    - input_file (dict): Input file containing student response, prompt, and question.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response, prompt, question, and current timestamp
        student_response = input_file["transcript"]
        prompt = input_file["metadata"]["prompt"]
        question = input_file["metadata"]["question"]
        current_time = datetime.datetime.now()

        input_file["data"] = {}

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i forgot"]
        
        # Check if the response contains "I don't know" variations
        text_is_idk = idk(idk_list, student_response)

        if (
            student_response is None
            or text_is_idk
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 3
        ):
            # Handle cases where response is invalid or contains "I don't know"
            input_file["score"] = 0
            input_file["data"]["grammar_check_output"] = ""
            input_file["data"]["grammatical_mistakes"] = 0
            input_file["data"]["corrections_made"] = None
            input_file["data"]["Spelling_errors"] = None
            input_file["data"]["total_number_of_spell_errors"] = None
        else:
            # Check for question or prompt copying, future tense, grammar correction, and spelling errors
            copied_from_question = check_question_copying(question, student_response)
            copied_from_prompt = check_prompt_copying(prompt, student_response)
            future_tense = futuretense(student_response)

            gramformer_corrected = list(gf.correct(student_response, max_candidates=1))[0]
            ginger_it_corrected = str(gramformer_corrected)
            original_vs_corrected = gf.get_edits(student_response, ginger_it_corrected)
            corrections = [corrections[1::3] for corrections in original_vs_corrected]

            # Filter spelling errors
            spelling_errors = filter_spell_errors(corrections)

            # Check sentence validity
            sentence_validity = check_validity(student_response)

            if (
                copied_from_prompt
                or copied_from_question
                or sentence_validity == "invalid"
                or sensible(student_response) is False
            ):
                # Handle cases where there is copying, invalid sentence, or non-sensible content
                input_file["score"] = 0
                input_file["data"]["grammar_check_output"] = ginger_it_corrected
                input_file["data"]["grammatical_mistakes"] = len(original_vs_corrected)
                input_file["data"]["corrections_made"] = [(corrections[1::3]) for corrections in original_vs_corrected]
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)
            elif (
                future_tense
                and len(student_response.split()) >= 3
                and len(spelling_errors) < 3
                and not copied_from_prompt
                and not copied_from_question
            ):
                # Handle cases where future tense is used, response is sensible, and there are minimal spelling errors
                input_file["score"] = 2
                input_file["data"]["grammar_check_output"] = ginger_it_corrected
                input_file["data"]["grammatical_mistakes"] = len(original_vs_corrected)
                input_file["data"]["corrections_made"] = [(corrections[1::3]) for corrections in original_vs_corrected]
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)
            else:
                # Handle cases where response is sensible with moderate spelling errors
                input_file["score"] = 1
                input_file["data"]["grammar_check_output"] = ginger_it_corrected
                input_file["data"]["grammatical_mistakes"] = len(original_vs_corrected)
                input_file["data"]["corrections_made"] = [(corrections[1::3]) for corrections in original_vs_corrected]
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output


def wtype3(input_file):
    """
    Perform autograding for a specific writing type (Type 3).

    Args:
    - input_file (dict): Input file containing student response, prompt, and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response, prompt, and keywords
        student_response = input_file["transcript"]
        prompt = input_file["metadata"]["prompt"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", " i donot know", "i forgot"]

        # Check for question copying, "I don't know" variations, and gibberish
        copied_from_prompt = check_prompt_copying(prompt, student_response)
        text_is_idk = idk(idk_list, student_response)

        input_file["data"] = {}

        if (
            student_response is None
            or text_is_idk
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 3
            or copied_from_prompt
        ):
            # Handle cases where response is invalid or contains "I don't know"
            input_file["score"] = 0
            input_file["data"]["grammar_check_output"] = ""
            input_file["data"]["grammatical_mistakes"] = 0
            input_file["data"]["corrections_made"] = []
            input_file["data"]["Spelling_errors"] = None
            input_file["data"]["total_number_of_spell_errors"] = None

        else:
            # Check grammar, spelling errors, invalid words, keyword presence, and sentence count
            grammar_check = grammar_check_gramformer(student_response)
            gram_output = grammar_check[0]
            corrections = grammar_check[1]
            total_gram_errors = len(corrections)

            # Filter spelling errors
            spelling_errors = filter_spell_errors(corrections)

            # Count invalid words
            number_of_invalid_words = invalid_words(student_response)

            # Check for the presence of keywords
            keyword_present = kw_presence(keywords, student_response)

            # Count the number of sentences
            number_of_sentence = count_sentence(student_response)

            if (
                number_of_sentence >= 3
                and keyword_present
                and len(spelling_errors) < 4
                and number_of_invalid_words < 4
            ):
                # Handle cases where conditions are met for a score of 4
                input_file["score"] = 4
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            elif (
                number_of_sentence >= 2
                and keyword_present
                and len(spelling_errors) < 4
                and number_of_invalid_words < 4
            ):
                # Handle cases where conditions are met for a score of 3
                input_file["score"] = 3
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            elif (
                number_of_sentence >= 2
                and keyword_present
                and len(spelling_errors) < 6
                and number_of_invalid_words < 5
                and len(student_response.split()) >= 4
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            elif (
                number_of_sentence >= 1
                and keyword_present
                and number_of_invalid_words < 5
                and len(student_response.split()) >= 3
                and total_gram_errors > 4
            ):
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            else:
                # Handle cases where none of the conditions are met
                if number_of_sentence >= 1 and keyword_present:
                    input_file["score"] = 1
                    input_file["data"]["grammar_check_output"] = gram_output
                    input_file["data"]["grammatical_mistakes"] = total_gram_errors
                    input_file["data"]["corrections_made"] = corrections
                    input_file["data"]["Spelling_errors"] = spelling_errors
                    input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)
                else:
                    input_file["score"] = 0
                    input_file["data"]["grammar_check_output"] = gram_output
                    input_file["data"]["grammatical_mistakes"] = total_gram_errors
                    input_file["data"]["corrections_made"] = corrections
                    input_file["data"]["Spelling_errors"] = spelling_errors
                    input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output


def wtype4(input_file):
    """
    Perform autograding for a specific writing type (Type 4).

    Args:
    - input_file (dict): Input file containing student response, prompt, and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response, prompt, and keywords
        student_response = input_file["transcript"]
        prompt = input_file["metadata"]["prompt"]
        keywords = input_file["metadata"]["keywords"]

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for question copying, "I don't know" variations, and gibberish
        copied_from_prompt = check_prompt_copying(prompt, student_response)
        text_is_idk = idk(idk_list, student_response)

        if (
            student_response is None
            or text_is_idk
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 3
            or copied_from_prompt
        ):
            # Handle cases where response is invalid or contains "I don't know"
            input_file["score"] = 0
            input_file["data"]["grammar_check_output"] = ""
            input_file["data"]["grammatical_mistakes"] = 0
            input_file["data"]["corrections_made"] = []
            input_file["data"]["Spelling_errors"] = None
            input_file["data"]["total_number_of_spell_errors"] = None

        else:
            # Check grammar, spelling errors, invalid words, keyword presence, sentence count,
            # sentence completeness, and compound sentence presence
            grammar_check = grammar_check_gramformer(student_response)
            gram_output = grammar_check[0]
            corrections = grammar_check[1]
            total_gram_errors = len(corrections)

            # Filter spelling errors
            spelling_errors = filter_spell_errors(corrections)

            # Count invalid words
            number_of_invalid_words = invalid_words(student_response)

            # Check for the presence of keywords
            keyword_present = kw_presence(keywords, student_response)

            # Count the number of sentences
            number_of_sentence = count_sentence(student_response)

            # Check sentence completeness and compound sentence presence
            complete_sentence = is_complete_sentence(student_response)
            conjuction_sentence = conjuction(student_response)

            if (
                number_of_sentence >= 1
                and conjuction_sentence
                and keyword_present
                and len(spelling_errors) < 3
                and complete_sentence
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            elif (
                number_of_sentence > 1
                and keyword_present
                and (2 < len(spelling_errors) < 7)
                and number_of_invalid_words < 6
                and len(student_response.split()) >= 4
            ):
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            else:
                # Handle cases where none of the conditions are met
                input_file["score"] = 1
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output


def wtype5(input_file):
    """
    Perform autograding for a specific writing type (Type 5).

    Args:
    - input_file (dict): Input file containing student response, prompt, and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response, prompt, and keywords
        student_response = input_file["transcript"]
        prompt = input_file["metadata"]["prompt"]
        keywords = input_file["metadata"]["keywords"]

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for question copying, "I don't know" variations, and gibberish
        copied_from_prompt = check_prompt_copying(prompt, student_response)
        text_is_idk = idk(idk_list, student_response)

        if (
            student_response is None
            or text_is_idk
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 3
            or copied_from_prompt
        ):
            # Handle cases where response is invalid or contains "I don't know"
            input_file["score"] = 0
            input_file["data"]["grammar_check_output"] = ""
            input_file["data"]["grammatical_mistakes"] = 0
            input_file["data"]["corrections_made"] = []
            input_file["data"]["Spelling_errors"] = None
            input_file["data"]["total_number_of_spell_errors"] = None

        else:
            # Check grammar, spelling errors, invalid words, keyword presence, sentence count,
            # sentence completeness, and compound sentence presence
            grammar_check = grammar_check_gramformer(student_response)
            gram_output = grammar_check[0]
            corrections = grammar_check[1]
            total_gram_errors = len(corrections)

            # Filter spelling errors
            spelling_errors = filter_spell_errors(corrections)

            # Count invalid words
            number_of_invalid_words = invalid_words(student_response)

            # Check for the presence of keywords
            keyword_present = kw_presence_Q5(keywords, student_response)

            # Count the number of sentences
            number_of_sentence = count_sentence(student_response)

            # Check sentence completeness and compound sentence presence
            complete_sentence = is_complete_sentence(student_response)
            conjuction(student_response)

            if (
                number_of_sentence >= 3
                and keyword_present
                and complete_sentence
                and len(spelling_errors) < 3
            ):
                # Handle cases where conditions are met for a score of 3
                input_file["score"] = 3
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            elif (
                number_of_sentence == 2
                and keyword_present
                and len(spelling_errors) < 5
                and number_of_invalid_words < 6
                and len(student_response.split()) >= 4
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            elif (
                number_of_sentence == 1
                and keyword_present
                and (3 < len(spelling_errors) < 9)
                and len(student_response.split()) >= 4
                or number_of_invalid_words > 7
            ):
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            else:
                # Handle cases where none of the conditions are met
                if keyword_present:
                    input_file["score"] = 1
                    input_file["data"]["grammar_check_output"] = gram_output
                    input_file["data"]["grammatical_mistakes"] = total_gram_errors
                    input_file["data"]["corrections_made"] = corrections
                    input_file["data"]["Spelling_errors"] = spelling_errors
                    input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)
                else:
                    input_file["score"] = 0
                    input_file["data"]["grammar_check_output"] = gram_output
                    input_file["data"]["grammatical_mistakes"] = total_gram_errors
                    input_file["data"]["corrections_made"] = corrections
                    input_file["data"]["Spelling_errors"] = spelling_errors
                    input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output



def wtype6(input_file):
    """
    Perform autograding for a specific writing type (Type 6).

    Args:
    - input_file (dict): Input file containing student response, prompt, and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response, prompt, and ideal answer
        student_response = input_file["transcript"]
        prompt = input_file["metadata"]["question"]
        keywords = input_file["metadata"]["keywords"]

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for question copying, "I don't know" variations, and gibberish
        copied_from_prompt = check_prompt_copying(prompt, student_response)
        text_is_idk = idk(idk_list, student_response)

        if (
            student_response is None
            or text_is_idk
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 5
            or copied_from_prompt
        ):
            # Handle cases where response is invalid or contains "I don't know"
            input_file["score"] = 0
            input_file["data"]["grammar_check_output"] = ""
            input_file["data"]["grammatical_mistakes"] = 0
            input_file["data"]["corrections_made"] = []
            input_file["data"]["Spelling_errors"] = None
            input_file["data"]["total_number_of_spell_errors"] = None

        else:
            # Check grammar, spelling errors, invalid words, keyword matches, sentence count,
            # sentence completeness, and compound sentence presence
            grammar_check = grammar_check_gramformer(student_response)
            gram_output = grammar_check[0]
            corrections = grammar_check[1]
            total_gram_errors = len(corrections)

            # Filter spelling errors
            spelling_errors = filter_spell_errors(corrections)

            # Count invalid words
            number_of_invalid_words = invalid_words(student_response)

            # Check for keyword matches
            keyword_matches = kw_presence_Q6(keywords, student_response)

            # Count the number of sentences
            number_of_sentence = count_sentence(student_response)

            # Check sentence completeness and compound sentence presence
            complete_sentence = is_complete_sentence(student_response)
            compound_sentence = conjuction(student_response)

            if (
                number_of_sentence >= 5
                and len(keyword_matches) > len(keywords.replace(' ','').split(','))//2
                and complete_sentence
                and len(spelling_errors) < 4
            ):
                # Handle cases where conditions are met for a score of 4
                input_file["score"] = 4
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            elif (
                number_of_sentence >= 3
                and len(keyword_matches) > len(keywords.replace(' ','').split(','))/3
                and len(spelling_errors) < 6
                and number_of_invalid_words <= 3
                and total_gram_errors < 3
            ):
                # Handle cases where conditions are met for a score of 3
                input_file["score"] = 3
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            elif (
                (number_of_sentence >= 2 or (number_of_sentence == 1 and compound_sentence))
                and len(keyword_matches) > 2
                and len(spelling_errors) < 7
                and number_of_invalid_words < 6
                and total_gram_errors < 6
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            elif (
                (number_of_sentence >= 2 or (number_of_sentence == 1 and compound_sentence))
                and len(keyword_matches) > 2
                and len(spelling_errors) < 6
                and number_of_invalid_words < 6
                and total_gram_errors < 7
            ):
                # Handle cases where conditions are met for a score of 2 (alternative condition)
                input_file["score"] = 2
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            elif (
                ((number_of_sentence == 2 or (number_of_sentence == 1 and compound_sentence))
                and len(keyword_matches) > 1
                and len(spelling_errors) < 7
                and len(student_response.split()) >= 4)
                or number_of_invalid_words > 6
            ):
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["grammar_check_output"] = gram_output
                input_file["data"]["grammatical_mistakes"] = total_gram_errors
                input_file["data"]["corrections_made"] = corrections
                input_file["data"]["Spelling_errors"] = spelling_errors
                input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

            else:
                # Handle cases where none of the conditions are met
                if len(keyword_matches) >= 2:
                    input_file["score"] = 1
                    input_file["data"]["grammar_check_output"] = gram_output
                    input_file["data"]["grammatical_mistakes"] = total_gram_errors
                    input_file["data"]["corrections_made"] = corrections
                    input_file["data"]["Spelling_errors"] = spelling_errors
                    input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)
                else:
                    input_file["score"] = 0
                    input_file["data"]["grammar_check_output"] = gram_output
                    input_file["data"]["grammatical_mistakes"] = total_gram_errors
                    input_file["data"]["corrections_made"] = corrections
                    input_file["data"]["Spelling_errors"] = spelling_errors
                    input_file["data"]["total_number_of_spell_errors"] = len(spelling_errors)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output

