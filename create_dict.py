import global_options
import pandas as pd
import pickle
from collections import OrderedDict, Counter
import itertools
from pprint import pprint
import gensim
from culture import culture_dictionary
from pathlib import Path

def load_w2v(w2v_path):
    """
    Load word2vec model

    Args:
        w2v_path (str): path of word2vec model

    Returns:
        model: word2vec model
    """
    print('Loading word2vec model...')
    model = gensim.models.KeyedVectors.load(w2v_path)
    return model

model = gensim.models.KeyedVectors.load(str(Path(global_options.MODEL_FOLDER, "w2v", "w2v.mod")))
#model = load_w2v(w2v_path = 'D:/Stanford_CoreNLP/wordsvec-mda-200-6/mda01-21.200.6.bin')
#model = gensim.models.KeyedVectors.load_word2vec_format('D:/Stanford_CoreNLP/tencent_zh_d200_small.txt', binary=False)
vocab_number = len(model.wv)

print("Vocab size in the w2v model: {}".format(vocab_number))

# expand dictionary
expanded_words = culture_dictionary.expand_words_dimension_mean(
    word2vec_model=model,
    seed_words=global_options.SEED_WORDS,
    restrict=global_options.DICT_RESTRICT_VOCAB,
    n=global_options.N_WORDS_DIM,
)
print("Dictionary created. ")
# make sure that one word only loads to one dimension
expanded_words = culture_dictionary.deduplicate_keywords(
    word2vec_model=model,
    expanded_words=expanded_words,
    seed_words=global_options.SEED_WORDS,
)
print("Dictionary deduplicated. ")

# rank the words under each dimension by similarity to the seed words
expanded_words = culture_dictionary.rank_by_sim(
    expanded_words, global_options.SEED_WORDS, model
)
# output the dictionary
culture_dictionary.write_dict_to_csv(
    culture_dict=expanded_words,
    file_name=str(Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv")),
)
print("Dictionary saved at {}".format(str(Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv"))))
