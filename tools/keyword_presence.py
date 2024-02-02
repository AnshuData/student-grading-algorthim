
from fuzzywuzzy import fuzz, process

def kw_presence(kws, stu_response) :
    """ checks if keywords are present in student_response or not """
    kw_present = False
    kws_matches = []   
    kws = kws.replace(' ','').split(',')
    seen_matches = set()
    
    try:
        tokens = stu_response.split()
        
        for kw in kws:
            matches = process.extract(kw, tokens, scorer=fuzz.token_set_ratio)
            best_match, score = max(matches, key=lambda x: x[1])
            if score > 80 and (best_match, kw) not in seen_matches:
                kws_matches.append((best_match))
                seen_matches.add((best_match, kw))
                
    except (TypeError, AttributeError):
        pass

    if len(kws_matches) > 1 :
        kw_present = True
    
    
    return kw_present