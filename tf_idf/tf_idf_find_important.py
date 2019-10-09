import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('inaugural')
from nltk.corpus import stopwords, brown, gutenberg, webtext, inaugural
import os
import argparse
from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from itertools import chain
from collections import Counter
from nltk.text import TextCollection
import numpy as np
import pandas as pd

class Sentence:
    """
    A class used to manipulate sentence.
    """
    word_tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = nltk.WordNetLemmatizer()
    stopwords_list = stopwords.words("english")

    def __init__(self, raw_sent):
        """
        Construct a Sentence object by given a raw text string
        :param raw_sent:  string of raw text
        """
        self.raw_sent = raw_sent

        lower_sent = self.raw_sent.lower()

        self.words = self.word_tokenizer.tokenize(lower_sent)

        # lemmatizer cost too much time, so we give it up here for now.
        # self.words = [self.lemmatizer.lemmatize(w) for w in self.words]

        # remove stopwords
        self.words = [w for w in self.words if w not in self.stopwords_list]

        # tag the words
        self.pos_tags = nltk.pos_tag(self.words)

    def get_parts_of_speech(self, parts_of_speech):
        """
        Return the words with certain part_of_speech
        :param parts_of_speech:  List of part of speech we want.
        :return:                 The words we need.
        """
        return [w for w, t in self.pos_tags if t in parts_of_speech]


class Document:
    """
    A class used to manipulate document
    """
    def __init__(self, file_name):
        """
        Construct a Document object with a filename
        :param file_name:  The path and name of the file
        """
        self.doc_name = Path(file_name).parts[-1]
        with open(file_name) as fp:
            self.raw_text = fp.read()
        self.preprocess()

    def preprocess(self):
        """
        Preprocess the raw text sentence by sentence
        :return:  None
        """
        self.sents = [Sentence(s) for s in nltk.sent_tokenize(self.raw_text)]

    def get_parts_of_speech(self, parts_of_speech: list=('NN', 'NNS', 'NNP', 'NNPS')):
        """
        Get the list of words by specifying certain types of part_of_speech
        :param parts_of_speech:
        :return:
        """
        out = [s.get_parts_of_speech(parts_of_speech) for s in self.sents]
        return list(chain(*out))


    def find_word(self, word):
        """
        Find a word in the document
        :param word:  The word to find.
        :return:      name of the document, sentences that contains the word
        """
        sent_with_word = []
        for s in self.sents:
            if word in s.words:
                sent_with_word.append(s.raw_sent)
        return self.doc_name, sent_with_word


def calculate_tf(documents):
    """
    Calculate the tf of words in documents
    :param documents:  List of documents
    :return:           counter, tf (normalized)
    """
    words = list(chain(*[doc.get_parts_of_speech() for doc in documents]))
    counter = Counter(words)

    # normalize
    tf = {k: v / len(words) for k, v in counter.items()}

    # counter contains the raw number
    # tf contains the normalized tf
    return counter, tf


def calculate_idf(words, corpus):
    """
    Calulate the idf of words by using a corpus
    :param words:  The words to calculate their idf
    :param corpus: The corpus to use in calculation
    :return:       dict of {word: idf}
    """
    words = set(words)
    # print("Loading corpus to calculate idf...")
    corpus_colleciton = TextCollection(corpus)
    idfs = {}
    for word in words:
        idfs[word] = corpus_colleciton.idf(word)
    return idfs


def show_words(words, documents):
    """
    Use pandas to show the important words
    :param words:     list of words to show
    :param documents: list of documents
    :return:          None
    """
    output_df = pd.DataFrame(columns=['Word', 'Documents', 'Sentences'])

    for word in words:
        for doc in documents:
            d, s = doc.find_word(word)
            if len(s) > 0:
                for i in s:
                    output_df = output_df.append({'Word': word, 'Documents': d, 'Sentences': i}, ignore_index=True)

    output_df = output_df.set_index(['Word', 'Documents'])
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', -1):
        print(output_df)


def parse_args():
    """
    This function is used to parse the command line arguments
    :return args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'PATH_NAME',
        help="Path of the file/folder."
    )
    parser.add_argument(
        'NUM_OF_IMPORTANT_WORDS',
        help="How many important words to show."
    )

    parser.add_argument(
        '--tf_lower_limit',
        default=2,
        help="The lower limit of words' tf."
    )

    return parser.parse_args()


if __name__ == '__main__':

    # get arguments from command line
    args = parse_args()


    # check filepath
    file_path = Path(args.PATH_NAME)
    if not file_path.exists():
        # raise error when the file or directory does not exist.
        raise FileNotFoundError(f"{str(file_path)} does not exist.")

    documents = [] # list to store documents
    if file_path.is_dir():
        # when the path is a directory, walk through the folder and make Document objects
        for dirpath, dirs, files in os.walk(str(file_path)):
            for f in files:
                fname = os.path.join(dirpath, f)
                documents.append(Document(fname))

    if file_path.is_file():
        # when the path is a file, simply build one Document object and store it in documents list
        documents.append(Document(str(file_path)))

    # calculate terms' frequency in theses documents
    counter, tf = calculate_tf(documents)

    # get the words with tf higher than the lower limit.
    words_to_calculate = [w for w, c in counter.items() if c > int(args.tf_lower_limit)]

    # calculate the idf of these words
    idf = calculate_idf(words_to_calculate, inaugural)

    # calculate the tf_idf values
    tf_idf = {k: tf[k] * v for k, v in idf.items()}
    # tf_idf = tf

    # sort the words by tf-idf values
    sorted_tf_idf = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
    words_to_show = sorted_tf_idf[:int(args.NUM_OF_IMPORTANT_WORDS)]

    # get the words to show
    words_to_show, _ = list(zip(*words_to_show))

    # show the important words
    show_words(words_to_show, documents)









