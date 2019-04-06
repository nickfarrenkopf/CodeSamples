import os
import gensim.models.word2vec as w2v

import data_things


class SmartSearch(object):
    """ comments and things """

    def __init__(self, wbed_name, stop_words):
        """ more comments """
        self.wbed = data_things.load_embedding(base_path, wbed_name)
        self.stop_words = stop_words
        self.vocab = self.wbed.wv.vocab


    ### WORD EMBEDDING ###

    # --- function called from outside --- #
    def phrase_check(self, input_text, phrases):
        """ tests given phrase against all test phrases """
        input_phrase = self.clean_user_input(input_text)
        if len(input_phrase) > 0:
            data_comparison = (input_text, ' '.join(input_phrase))
            top_phrases = self.get_similar(input_phrase, phrases)
            return (data_comparison, top_phrases)
        return ((input_text, ''), [])

    def get_similar(self, user_phrase, phrases, top_n=15):
        """ returns sentences silimar to user typed phrase """
        print([user_phrase])
        print(phrases[0][0].split(' '))
        metrics = [('{:.5f}'.format(self.wbed.wv.n_similarity(user_phrase, p[0].split(' '))),
                    p[1]) for p in phrases]
        metrics = list(reversed(sorted(metrics, key=lambda x:x[0])))
        top_values = metrics[:top_n]
        return top_values


    ### HELPER ###

    def clean_user_input(self, input_text):
        """ cleans user input against stop words and vocab list """
        clean_text = data_things.clean(input_text)
        input_phrase = [word for word in clean_text.split(' ')
                        if word in self.vocab and word not in self.stop_words]
        while None in input_phrase:
            input_phrase.remove(None)
        return input_phrase

    def clean_phrase(self, phrase):
        """ """
        phrase = ' '.join([p for p in data_things.clean(phrase).split(' ')
                           if p in self.vocab])
        return phrase


### PARAMS ###

# location
base_path = os.path.dirname(os.path.realpath(__file__))
stop_path = os.path.join(base_path, 'stop_words_16.txt')

# wbed
wbed_name = 'general_300_embedding'
stop_words = data_things.get_stop_words(stop_path)

# smart search
ss = SmartSearch(wbed_name, stop_words)


