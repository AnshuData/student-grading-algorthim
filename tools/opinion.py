
def opinion_justified(answer):
    opinion_options = ["think", "because", "opinion", "better", "option", "since", "believe" , "best"]

    justified = False

    for options in opinion_options:
        if options in answer.lower():
            words = answer.lower().split()
            index = words.index(options)+1
            if len(words)-index > 6:
                justified = True
                break
            
    return justified