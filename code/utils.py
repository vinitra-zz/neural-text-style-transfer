import pandas as pd
import numpy as np
import json
from collections import defaultdict

def import_dataset():
        with open('../data/authorship_corpora/influencer_tweets.json', 'r') as fp:
                dataset = json.load(fp)
                
        text = []
        user = []

        for t in dataset:
            text.append(t['text'])
            user.append(t['user'])
        
        data = pd.DataFrame.from_dict({"text":text, "author":user})
        
        print("Loading {} tweets from {} unique users.".format(len(data), len(set(data['author']))))
        return data

def load_charset():
    """
    How the character set was created from the corpus
    # preprocessing to get the character set
    from collections import defaultdict
    import json

    tweets = data.text
    char_count = defaultdict(int)

    for t in tqdm(data.text.str.lower()):
        for c in t:
            char_count[c]+=1


    with open('../outputs/charset.json', 'w') as outfile:
        json.dump(char_count, outfile)
    """

    f = open('../outputs/charset.json', 'r')
    charset = json.load(f)


    return charset



