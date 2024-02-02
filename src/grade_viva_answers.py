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






def stype1(input_file):
    """
    Perform autograding for a specific speaking type (Type 1).

    Args:
    - input_file (dict): Input file containing student response and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response and keywords from input
        student_response = input_file["transcript"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for "I don't know" variations and gibberish
        text_is_idk = idk(idk_list, student_response)

        # Check for keyword presence in spoken response
        keyword_present = kw_presence_spoken(keywords, student_response)

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        if (
            student_response is None
            or (text_is_idk and len(student_response.split()) < 5)
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 4
            or len(keyword_present) == 0
        ):
            # Handle cases where response is invalid or does not meet conditions
            input_file["score"] = 0
            input_file["data"]["keyword_present"] = "None"
        else:
            # Handle cases where conditions are met for a score of 1
            input_file["score"] = 1
            input_file["data"]["keyword_present"] = ",".join(keyword_present)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output




def stype2(input_file):
    """
    Perform autograding for a specific speaking type (Type 2).

    Args:
    - input_file (dict): Input file containing student response and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response and keywords from input
        student_response = input_file["transcript"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for "I don't know" variations and gibberish
        text_is_idk = idk(idk_list, student_response)

        # Check for keyword presence in spoken response
        keyword_present = kw_presence_spoken(keywords, student_response)

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        if (
            student_response is None
            or (text_is_idk and len(student_response.split()) < 5)
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 4
            or len(keyword_present) == 0
        ):
            # Handle cases where response is invalid or does not meet conditions
            input_file["score"] = 0
            input_file["data"]["keyword_present"] = "None"
        else:
            # Handle cases where conditions are met for a score of 1
            input_file["score"] = 1
            input_file["data"]["keyword_present"] = ",".join(keyword_present)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output




def stype3(input_file):
    """
    Perform autograding for a specific speaking type (Type 3).

    Args:
    - input_file (dict): Input file containing student response and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response and keywords from input
        student_response = input_file["transcript"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for "I don't know" variations and gibberish
        text_is_idk = idk(idk_list, student_response)

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        if (
            student_response is None
            or (text_is_idk and len(student_response.split()) < 5)
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 2
        ):
            # Handle cases where response is invalid or does not meet conditions
            input_file["score"] = 0
            input_file["data"]["keyword_present"] = "None"
        else:
            # Check for complete sentence, spell errors, keyword presence, and sentence length
            complete_sentence = is_complete_sentence(student_response)
            keyword_present = kw_presence_spoken(keywords, student_response)
            sentence_length = count_sentence(student_response)
            errors = grammar_check_gramformer(student_response)[1]
            spell_errors = len(filter_spell_errors(errors))

            if (
                complete_sentence
                and spell_errors < 6
                and len(keyword_present) >= 1
                and sentence_length >= 1
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif len(keyword_present) >= 1:
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif complete_sentence and len(student_response.split()) > 5:
                # Handle additional cases for a score of 1
                input_file["score"] = 1
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            else:
                # Handle cases where conditions are not met for a score
                input_file["score"] = 0
                input_file["data"]["keyword_present"] = ",".join(keyword_present)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output


import json

def stype4(input_file):
    """
    Perform autograding for a specific speaking type (Type 4).

    Args:
    - input_file (dict): Input file containing student response and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response and keywords from input
        student_response = input_file["transcript"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for "I don't know" variations and gibberish
        text_is_idk = idk(idk_list, student_response)

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        if (
            student_response is None
            or (text_is_idk and len(student_response.split()) < 5)
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 2
        ):
            # Handle cases where response is invalid or does not meet conditions
            input_file["score"] = 0
            input_file["data"]["keyword_present"] = "None"
        else:
            # Check for complete sentence, spell errors, keyword presence, and sentence length
            complete_sentence = is_complete_sentence(student_response)
            keyword_present = kw_presence_spoken(keywords, student_response)
            sentence_length = count_sentence(student_response)
            errors = grammar_check_gramformer(student_response)[1]
            spell_errors = len(filter_spell_errors(errors))

            if (
                complete_sentence
                and spell_errors < 6
                and len(keyword_present) >= 1
                and sentence_length >= 1
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif len(keyword_present) >= 1:
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            else:
                # Handle cases where conditions are not met for a score
                input_file["score"] = 0
                input_file["data"]["keyword_present"] = ",".join(keyword_present)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output


def stype5(input_file):
    """
    Perform autograding for a specific speaking type (Type 5).

    Args:
    - input_file (dict): Input file containing student response and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response and keywords from input
        student_response = input_file["transcript"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for "I don't know" variations and gibberish
        text_is_idk = idk(idk_list, student_response)

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        if (
            student_response is None
            or (text_is_idk and len(student_response.split()) < 5)
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 3
        ):
            # Handle cases where response is invalid or does not meet conditions
            input_file["score"] = 0
            input_file["data"]["keyword_present"] = "None"
        else:
            # Check for complete sentence, spell errors, keyword presence, sentence length, and future tense
            complete_sentence = is_complete_sentence(student_response)
            keyword_present = kw_presence_spoken(keywords, student_response)
            sentence_length = count_sentence(student_response)
            errors = grammar_check_gramformer(student_response)[1]
            spell_errors = len(filter_spell_errors(errors))

            if (
                complete_sentence
                and spell_errors < 6
                and len(keyword_present) >= 1
                and sentence_length >= 1
                and ("?" in student_response or futuretense(student_response) == True)
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            else:
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["keyword_present"] = ",".join(keyword_present)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output


def stype6(input_file):
    """
    Perform autograding for a specific speaking type (Type 6).

    Args:
    - input_file (dict): Input file containing student response and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response and keywords from input
        student_response = input_file["transcript"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for "I don't know" variations and gibberish
        text_is_idk = idk(idk_list, student_response)

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        if (
            student_response is None
            or (text_is_idk and len(student_response.split()) < 5)
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 3
        ):
            # Handle cases where response is invalid or does not meet conditions
            input_file["score"] = 0
            input_file["data"]["keyword_present"] = "None"
        else:
            # Check for complete sentence, spell errors, keyword presence, sentence length, and future tense
            complete_sentence = is_complete_sentence(student_response)
            keyword_present = kw_presence_spoken(keywords, student_response)
            sentence_length = count_sentence(student_response)
            errors = grammar_check_gramformer(student_response)[1]
            spell_errors = len(filter_spell_errors(errors))

            if (
                complete_sentence
                and spell_errors < 6
                and len(keyword_present) >= 1
                and sentence_length >= 1
                and ("?" in student_response or futuretense(student_response) == True)
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            else:
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["keyword_present"] = ",".join(keyword_present)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output


import json

def stype7(input_file):
    """
    Perform autograding for a specific speaking type (Type 7).

    Args:
    - input_file (dict): Input file containing student response and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response and keywords from input
        student_response = input_file["transcript"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for "I don't know" variations and gibberish
        text_is_idk = idk(idk_list, student_response)

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        if (
            student_response is None
            or (text_is_idk and len(student_response.split()) < 5)
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 5
        ):
            # Handle cases where response is invalid or does not meet conditions
            input_file["score"] = 0
            input_file["data"]["keyword_present"] = "None"
        else:
            # Check for complete sentence, spell errors, keyword presence, sentence length, and opinion justification
            complete_sentence = is_complete_sentence(student_response)
            keyword_present = kw_presence_spoken(keywords, student_response)
            sentence_length = count_sentence(student_response)
            errors = grammar_check_gramformer(student_response)[1]
            spell_errors = len(filter_spell_errors(errors))
            opinion_made = opinion_justified(student_response)

            if (
                complete_sentence
                and spell_errors < 6
                and len(keyword_present) >= 2
                and len(student_response.split()) > 25
                and opinion_made
            ):
                # Handle cases where conditions are met for a score of 3
                input_file["score"] = 3
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                complete_sentence
                and len(keyword_present) >= 1
                and (len(student_response.split()) < 25 and sentence_length >= 3)
                and opinion_made
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                (len(keyword_present) >= 1 and sentence_length >= 1)
                or (len(keyword_present) >= 1 and len(student_response.split()) > 8)
            ):
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            else:
                # Handle cases where conditions are not met for a score
                input_file["score"] = 0
                input_file["data"]["keyword_present"] = ",".join(keyword_present)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output


def stype8(input_file):
    """
    Perform autograding for a specific speaking type (Type 8).

    Args:
    - input_file (dict): Input file containing student response and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response and keywords from input
        student_response = input_file["transcript"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for "I don't know" variations and gibberish
        text_is_idk = idk(idk_list, student_response)

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        if (
            student_response is None
            or (text_is_idk and len(student_response.split()) < 5)
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 5
        ):
            # Handle cases where response is invalid or does not meet conditions
            input_file["score"] = 0
            input_file["data"]["keyword_present"] = "None"
        else:
            # Check for complete sentence, spell errors, keyword presence, sentence length, and opinion justification
            complete_sentence = is_complete_sentence(student_response)
            keyword_present = kw_presence_spoken(keywords, student_response)
            sentence_length = count_sentence(student_response)
            errors = grammar_check_gramformer(student_response)[1]
            spell_errors = len(filter_spell_errors(errors))
            opinion_made = opinion_justified(student_response)

            if (
                complete_sentence
                and spell_errors < 10
                and len(keyword_present) >= 2
                and (
                    (sentence_length >= 3 and len(student_response.split()) < 25)
                    or len(student_response.split()) < 22
                )
                and opinion_made
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                complete_sentence
                and spell_errors < 6
                and len(keyword_present) >= 2
                and len(student_response.split()) > 25
                and opinion_made
            ):
                # Handle cases where conditions are met for a score of 3
                input_file["score"] = 3
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                (len(keyword_present) >= 1 and sentence_length >= 1)
                or (len(keyword_present) >= 1 and len(student_response.split()) > 8)
            ):
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            else:
                # Handle cases where conditions are not met for a score
                input_file["score"] = 0
                input_file["data"]["keyword_present"] = ",".join(keyword_present)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output


def stype9(input_file):
    """
    Perform autograding for a specific speaking type (Type 9).

    Args:
    - input_file (dict): Input file containing student response and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response and keywords from input
        student_response = input_file["transcript"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for "I don't know" variations and gibberish
        text_is_idk = idk(idk_list, student_response)

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        if (
            student_response is None
            or (text_is_idk and len(student_response.split()) < 5)
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 5
        ):
            # Handle cases where response is invalid or does not meet conditions
            input_file["score"] = 0
            input_file["data"]["keyword_present"] = "None"
        else:
            # Check for complete sentence, spell errors, keyword presence, sentence length
            complete_sentence = is_complete_sentence(student_response)
            keyword_present = kw_presence_spoken(keywords, student_response)
            sentence_length = count_sentence(student_response)
            errors = grammar_check_gramformer(student_response)[1]
            spell_errors = len(filter_spell_errors(errors))

            if (
                complete_sentence
                and spell_errors < 5
                and len(keyword_present) >= 3
                and sentence_length >= 2
                and len(student_response.split()) > 25
            ):
                # Handle cases where conditions are met for a score of 3
                input_file["score"] = 3
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                complete_sentence
                and spell_errors < 10
                and len(keyword_present) >= 2
                and len(student_response.split()) > 20
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                (len(keyword_present) >= 1 and sentence_length >= 1)
                or (len(keyword_present) >= 1 and len(student_response.split()) > 10)
            ):
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            else:
                # Handle cases where conditions are not met for a score
                input_file["score"] = 0
                input_file["data"]["keyword_present"] = ",".join(keyword_present)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output


import json

def stype10(input_file):
    """
    Perform autograding for a specific speaking type (Type 10).

    Args:
    - input_file (dict): Input file containing student response and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response and keywords from input
        student_response = input_file["transcript"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for "I don't know" variations and gibberish
        text_is_idk = idk(idk_list, student_response)

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        if (
            student_response is None
            or (text_is_idk and len(student_response.split()) < 5)
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 5
        ):
            # Handle cases where response is invalid or does not meet conditions
            input_file["score"] = 0
            input_file["data"]["keyword_present"] = "None"
        else:
            # Check for complete sentence, spell errors, keyword presence, sentence length
            complete_sentence = is_complete_sentence(student_response)
            keyword_present = kw_presence_spoken(keywords, student_response)
            sentence_length = count_sentence(student_response)
            errors = grammar_check_gramformer(student_response)[1]
            spell_errors = len(filter_spell_errors(errors))

            if (
                complete_sentence
                and spell_errors < 6
                and len(keyword_present) >= 2
                and sentence_length >= 2
                and len(student_response.split()) > 25
            ):
                # Handle cases where conditions are met for a score of 3
                input_file["score"] = 3
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                complete_sentence
                and spell_errors < 10
                and len(keyword_present) >= 2
                and len(student_response.split()) > 20
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                len(keyword_present) >= 1
                and len(student_response.split()) > 10
            ):
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            else:
                # Handle cases where conditions are not met for a score
                input_file["score"] = 0
                input_file["data"]["keyword_present"] = ",".join(keyword_present)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output


def stype11(input_file):
    """
    Perform autograding for a specific speaking type (Type 11).

    Args:
    - input_file (dict): Input file containing student response and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response and keywords from input
        student_response = input_file["transcript"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for "I don't know" variations and gibberish
        text_is_idk = idk(idk_list, student_response)

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        if (
            student_response is None
            or (text_is_idk and len(student_response.split()) < 5)
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 5
        ):
            # Handle cases where response is invalid or does not meet conditions
            input_file["score"] = 0
            input_file["data"]["keyword_present"] = "None"
        else:
            # Check for complete sentence, spell errors, keyword presence, sentence length
            complete_sentence = is_complete_sentence(student_response)
            keyword_present = kw_presence_spoken(keywords, student_response)
            sentence_length = count_sentence(student_response)
            errors = grammar_check_gramformer(student_response)[1]
            spell_errors = len(filter_spell_errors(errors))

            if (
                complete_sentence
                and spell_errors < 6
                and len(keyword_present) >= 3
                and sentence_length >= 4
            ):
                # Handle cases where conditions are met for a score of 4
                input_file["score"] = 4
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                complete_sentence
                and spell_errors < 10
                and len(keyword_present) >= 2
                and sentence_length >= 3
            ):
                # Handle cases where conditions are met for a score of 3
                input_file["score"] = 3
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                complete_sentence
                and spell_errors < 14
                and len(keyword_present) >= 2
                and sentence_length >= 3
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                complete_sentence
                and spell_errors < 16
                and len(keyword_present) >= 2
                and sentence_length >= 2
                and len(student_response.split()) > 12
            ):
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            else:
                # Handle cases where conditions are not met for a score
                if (
                    len(keyword_present) >= 1
                    and len(student_response.split()) > 12
                ):
                    input_file["score"] = 1
                    input_file["data"]["keyword_present"] = ",".join(keyword_present)
                else:
                    input_file["score"] = 0
                    input_file["data"]["keyword_present"] = ",".join(keyword_present)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output


def stype12(input_file):
    """
    Perform autograding for a specific speaking type (Type 12).

    Args:
    - input_file (dict): Input file containing student response and keywords.

    Returns:
    - str: JSON representation of the autograding results.
    """
    try:
        # Extract student response and keywords from input
        student_response = input_file["transcript"]
        keywords = input_file["metadata"]["keywords"]

        # List of "I don't know" variations
        idk_list = ["i don't know", "idk", "i donot know", "i forgot"]

        # Check for "I don't know" variations and gibberish
        text_is_idk = idk(idk_list, student_response)

        # Initialize data dictionary in input_file
        input_file["data"] = {}

        if (
            student_response is None
            or (text_is_idk and len(student_response.split()) < 5)
            or Detector.is_gibberish(student_response)
            or len(student_response.split()) < 5
        ):
            # Handle cases where response is invalid or does not meet conditions
            input_file["score"] = 0
            input_file["data"]["keyword_present"] = "None"
        else:
            # Check for complete sentence, spell errors, keyword presence, sentence length
            complete_sentence = is_complete_sentence(student_response)
            keyword_present = kw_presence_spoken(keywords, student_response)
            sentence_length = count_sentence(student_response)
            errors = grammar_check_gramformer(student_response)[1]
            spell_errors = len(filter_spell_errors(errors))

            if (
                complete_sentence
                and spell_errors < 6
                and len(keyword_present) >= 3
                and sentence_length >= 4
            ):
                # Handle cases where conditions are met for a score of 4
                input_file["score"] = 4
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                complete_sentence
                and spell_errors < 10
                and len(keyword_present) >= 2
                and sentence_length >= 3
            ):
                # Handle cases where conditions are met for a score of 3
                input_file["score"] = 3
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                complete_sentence
                and spell_errors < 14
                and len(keyword_present) >= 2
                and sentence_length >= 3
            ):
                # Handle cases where conditions are met for a score of 2
                input_file["score"] = 2
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            elif (
                complete_sentence
                and spell_errors < 16
                and len(keyword_present) >= 2
                and sentence_length >= 2
                and len(student_response.split()) > 12
            ):
                # Handle cases where conditions are met for a score of 1
                input_file["score"] = 1
                input_file["data"]["keyword_present"] = ",".join(keyword_present)
            else:
                # Handle cases where conditions are not met for a score
                if (
                    len(keyword_present) >= 1
                    and sentence_length >= 1
                    and len(student_response.split()) > 12
                ):
                    input_file["score"] = 1
                    input_file["data"]["keyword_present"] = ",".join(keyword_present)
                else:
                    input_file["score"] = 0
                    input_file["data"]["keyword_present"] = ",".join(keyword_present)

    except Exception as e:
        print(f"There was an exception: {e}")

    # Convert the result to JSON
    output = json.dumps(input_file)
    return output
