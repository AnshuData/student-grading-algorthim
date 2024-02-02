

def conjuction(answer):

    """ look for the conjuction/compound words in the sentence """

    conjuction_present = False
    conjuctions = ['and','or','then']

    for item in conjuctions:
        if f" {item} " in f" {answer} ":
            conjuction_present = True
            break
        else:
            conjuction_present = False
    #print(conjuction_present)   
    #    
    return conjuction_present

