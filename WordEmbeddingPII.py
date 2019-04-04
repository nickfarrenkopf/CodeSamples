import os
from scipy.spatial.distance import cosine
from gensim.models import word2vec as w2v

import data_things as dt


### CREATE ###

def create_wbed_1(size=100, min_count=10, win_size=5, alpha=0.025, iters=100):
    """ creates first embedding based on all text data """
    bigram, trigram, sents = dt.load_all_trans_data()
    sents = dt.token(sents)
    model = w2v.Word2Vec(sents, sg=1, size=size, min_count=min_count, 
                         window=win_size, alpha=alpha, iter=iters)
    model.save(paths.wbed_1_path)
    print('Word embedding 1 created')

def create_wbed_2(size=100, min_count=10, win_size=5, alpha=0.025, iters=100):
    """ creates second embedding based on all text data """
    bigram, trigram, sents = dt.load_all_trans_data()
    sents = dt.token(sents)
    model = w2v.Word2Vec(sents, sg=1, size=size, min_count=min_count, 
                         window=win_size, alpha=alpha, iter=iters)
    model.save(paths.wbed_2_path)
    print('Word embedding 2 created')

def create_wbed_t8(size=100, min_count=7, win_size=5, alpha=0.025, iters=100):
    """ """
    sentences = w2v.Text8Corpus(paths.text8_path)
    model = w2v.Word2Vec(sentences, size=size, sg=1, min_count=min_count,
                         window=win_size, alpha=0.025, iter=100)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=token_count, epochs=epochs)
    model.save(paths.text8_path)
    print('Text8 embedding created')


### TEST ###

def test_wbed(wbed, words, search_words, topn=15):
    """ runs word tests on word embeddings """
    _ = [most_similar(wbed, word, topn) for word in words]
    #_ = [compare(wbed, word, topn, search_words) for word in words]

def most_similar(wbed, word, topn):
    """ finds words most similar to input word """
    print('Similar to: {}'.format(word))
    for row in wbed.wv.most_similar(word, topn=topn):
        print('{} {}'.format(*row))
    print('')

def compare(wbed, word, topn, search_words):
    """ compare word to search words """
    print('Compare to: {}'.format(word))
    vecs = [(w, 1 - cosine(wbed[word], wbed[w])) for w in search_words
            if word in wbed and w in wbed and word != w]
    vecs = list(reversed(sorted(vecs, key=lambda x: x[1])))
    for row in vecs[:topn]:
        print('{} {}'.format(*row))
    print('')
    

### PROGRAM ###

if __name__ == '__main__':

    import paths

    # EMBEDDINGS
    # create_wbed_big()
    # create_wbed_small()
    if True:
        wbed_1 = dt.load_embedding(paths.wbed_1_path)
        wbed_2 = dt.load_embedding(paths.wbed_2_path)
        print('Len 1 vocab: {}'.format(len(wbed_1.wv.vocab)))
        print('Len 2 vocab: {}\n'.format(len(wbed_2.wv.vocab)))

    # TEST EMBEDDING
    search_words = paths.pii_blacklist
    ws1 = ['ted','chandler', 'andy','honey']
    ws2 = ['victoria','natalie','kuvira','tammy','larry','ellen','cindy']
    ws3 = ['red','green','coffee','office','book','fire','water']
    ws4 = ['oh', 'dude', 'guys', 'closure', 'ho_ho', 'girl']
    ws6 = ['ha_ha', 'fridge', 'nana', 'buffay', 'okay', 'ahem']
    ws7 = ['audition', 'yeah', 'traitor', 'fire_nation', 'kuvira', 'benders']
    
    ### TEST WBED 1 ###
    if True:
        test_wbed(wbed_1, ws3, search_words)

    ### TEST WBED 2 ###
    if False:
        test_wbed(wbed_2, ws1, search_words)


