import pandas as pd
import numpy as np
import json
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import load_model
import time
from keras.layers import InputLayer, Convolution1D, MaxPooling1D, Concatenate, Flatten, Dense, Dropout, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint

np.random.seed(123)

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

def load_10_people():

    """
    Load preprocessed dataset of 10 most prolific twitter authors in dataset.
    Return a dictionary with keys X_train, Y_train, X_val, Y_val, X_test, Y_test
    """

    np.random.seed(123)

    # Constants
    max_len_char = 140


    # load the dataset
    start = time.time()

    data = import_dataset()

    print('Loading Twitter dataset took %d seconds.' % (time.time() - start))

    # load the character set
    chars_set = load_charset()

    # Only keep the top 10 most frequent authors to work with
    top10_authors = np.array(data.author.value_counts().index[:10])
    top10_authors

    top10_authors_data = data[data.author.isin(top10_authors)]
    print("Number of Tweets: {}".format(len(top10_authors_data)))

    # only keep characters that appear at least 100 times in the corpus
    print("Only keeping characters that appear at least 100 times in the corpus")
    small_chars_set =  list(filter(lambda x: x[1]>=100, chars_set.items()))
    small_chars_set = list(sorted(small_chars_set, key = lambda x: x[0]))

    small_char_indices = dict((c[0], i) for i, c in enumerate(small_chars_set))

    print("Character set consists of {} characters".format(len(small_chars_set)))

    X_char = np.zeros((len(top10_authors_data), max_len_char, len(small_chars_set)), dtype=np.bool)
    # build the data matrix with the OHE stuff
    # padding is incorporated in the process by letting it be 0

    # building X
    print("Building X...")
    for doc_num, doc in enumerate(top10_authors_data.text):
        for char_num, char in enumerate(doc):
            # how to deal with docs, just keep part of the document             
            if char_num >= max_len_char:
                break
            # unknown characters and padding are all mapped to the 0 vector
            if char in small_char_indices:
                    X_char[doc_num, char_num, small_char_indices[char]] = 1

    # building Y
    print("Building Y...")

    ohe = OneHotEncoder()
    le = LabelEncoder()
    Y = ohe.fit_transform(le.fit_transform(top10_authors_data.author.values).reshape(-1, 1)).todense()

    # Train-Validation-Test split
    # Test is 90% of original
    # Val is 10% of train

    print("Splitting Data...")
    X_train_char, X_test_char, Y_train, Y_test = train_test_split(X_char, Y, test_size=0.10, random_state=42)

    X_train_char, X_val_char, Y_train, Y_val = train_test_split(X_train_char, Y_train, test_size=0.10, random_state=42)

    print('%d train char sequences' % len(X_train_char))
    print('%d test char sequences' % len(X_test_char))
    print('%d validation char sequences' % len(X_val_char))

    data = {
    "X_train": X_train_char,
    "Y_train": Y_train,
    "X_val": X_val_char,
    "Y_val": Y_val,
    "X_test": X_test_char,
    "Y_test": Y_test,
    }

    return data

# Load Models

def load_model_(filename, model_type="keras"):
    if model_type=="keras":
        model = load_model(filename)

    return model

def load_ruder_10_authors():
    model = load_model_("../models/ruder-09-0.97.hdf5")
    return model



