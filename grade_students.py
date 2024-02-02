# all imports

import datetime
import json

import nltk
import spacy.cli
import uvicorn
from gramformer import Gramformer
from fastapi import Body, FastAPI


from src.grade_viva_answers import stype1, stype2, stype3, stype4, stype5, stype6, stype7, stype8, stype9, stype10, stype11, stype12
from src.grade_written_answers import wtype1, wtype2, wtype3, wtype4, wtype5, wtype6


app = FastAPI()

# Sample endpoint
@app.get("/")
async def main():
    """
    Sample FastAPI endpoint.
    """
    print("Running Autograder")
    return "Running Autograder !!"


@app.post("/grade")
def autograding(json_input: str = Body()): 
    """
        Perform autograding based on the input JSON file.

        Args:
        - input_file (dict): JSON input containing metadata, question type, and responses.

        Returns:
        - dict: Autograding response.
    """
    
    input_file = json.loads(json_input) # read json input
    q_type = input_file["metadata"]["question_type"]
    section = input_file["section"]

    response = {}
    if(section == "Writing"):
        if(q_type == "Written_1_PictureDescription_GrammarCorrection"):
            response = wtype1(input_file)
        if(q_type == "Written_2_PictureDescription_WhatsNext"):
            response = wtype2(input_file)
        if(q_type == "Written_3_Experience"):
            response = wtype3(input_file)
        if(q_type == "Written_4_Academic_OneSentence"):
            response = wtype4(input_file)
        if(q_type == "Written_5_Academic_Paragraph"):
            response = wtype5(input_file)
        if(q_type == "Written_6_Opinion"):
            response = wtype6(input_file)
    if(section == "Speaking"):
        if(q_type == "Spoken_1_Picture_ShortAnswer"):
            response = stype1(input_file)
        if(q_type == "Spoken_2_Picture_ShortAnswer"):
            response = stype2(input_file)
        if(q_type == "Spoken_3_4_HowDoYouKnow"):
            response = stype3(input_file)
        if(q_type == "Spoken_3_4_ShortAnswer"):
            response = stype4(input_file)
        if(q_type == "Spoken_3_4_WhatNext"):
            response = stype5(input_file)
        if(q_type == "Spoken_5_6"):
            response = stype6(input_file)
        if(q_type == "Spoken_7_8_Opinion"):
            response = stype7(input_file)
        if(q_type == "Spoken_9_Academic_Description"):
            response = stype8(input_file)
        if(q_type == "Spoken_10_Academic_Question"):
            response = stype9(input_file)
        if(q_type == "Spoken_11_12_AcademicSummary"):
            response = stype10(input_file)

    return response




# sample call
# json_input='{"transcript":"a girl is using her laptop.","metadata":{"question_type":"jjj","ideal_answer":"A girl is using her laptop."}}'
# autograding(json_input)


if __name__ == "__main__":
    # Download necessary NLTK resources in the server running it
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    # Initialize Gramformer for grammar checking
    gf = Gramformer(models=1, use_gpu=False)

    # Run FastAPI app
    uvicorn.run("gram_check:app", host="0.0.0.0", port=80)




