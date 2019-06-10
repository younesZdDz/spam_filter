from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from collections import Counter
import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models.ldamulticore import LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000, column = 'whole'):
        self.vocabulary_size = vocabulary_size
        self.column = column
    def fit(self, X, y=None):
        counter = []
        for text in X[self.column].values:
            counter.append(Counter(text.split()))
        total_count = Counter()
        for word_count in counter:
            if isinstance(word_count, list):
                print(word_count)
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        counter = []
        for text in X[self.column].values:
            counter.append(Counter(text.split()))
            
        rows = []
        cols = []
        data = []        
        for row, word_count in enumerate(counter):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)        
        return pd.DataFrame(columns=['word_UNK'] + ['word_'+column for column in self.vocabulary_], 
                            data=csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1)).toarray())


class StructureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column = 'structure'):
        self.column = column
        
    def fit(self, X, y=None):     
        tmp = []
        for email in X[self.column].apply(lambda x : x.split(', ')).values :
            for structure in email:
                tmp.append(structure)        
        self.structures = list(Counter(tmp).keys())        
        return self
    
    def transform(self, X, y=None):
        out = np.zeros((len(X), len(self.structures)))
        for i , structure in enumerate(self.structures):
            out[:,i] = X[self.column].apply(lambda x : 1 if structure in x.split(', ') else 0).values
        return out

class LdaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dim = 2, column = 'whole'):
        self.dim = dim
        self.column = column
    def fit(self, X, y=None):     
        lda_tokens = X[self.column].apply(lambda x: x.split())
        # create Dictionary and train it on text corpus
        self.lda_dic = Dictionary(lda_tokens)
        self.lda_dic.filter_extremes(no_below=10, no_above=0.6, keep_n=8000)
        lda_corpus = [self.lda_dic.doc2bow(doc) for doc in lda_tokens]
        # create TfidfModel and train it on text corpus
        self.lda_tfidf = TfidfModel(lda_corpus)
        lda_corpus = self.lda_tfidf[lda_corpus]
        # create LDA Model and train it on text corpus
        self.lda_model = LdaMulticore(
            lda_corpus, num_topics=self.dim, id2word=self.lda_dic, workers=4,
            passes=20, chunksize=1000, random_state=0
        )
        return self
    
    def transform(self, X, y=None):
        lda_emb_len = len(self.lda_model[[]])
        lda_corpus = [self.lda_dic.doc2bow(doc) for doc in X[self.column].apply(lambda x: x.split())]
        lda_corpus = self.lda_tfidf[lda_corpus]
        lda_que_embs = self.lda_model.inference(lda_corpus)[0]
        # append lda question embeddings
        out = np.zeros((len(X), lda_emb_len))
        for i in range(lda_emb_len):
            out[:, i] = lda_que_embs[:, i]
        return out



class TfIdfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column = 'whole'):
        self.column = column
        self.model = TfidfVectorizer(lowercase = False, max_df=0.6, min_df=0.1, analyzer='char_wb', ngram_range=(1,3))
    def fit(self, X, y=None):     
        self.model = self.model.fit(X[self.column])
        return self
    
    def transform(self, X, y=None):
        self.model.transform(X[self.column])
        return self.model.transform(X[self.column])        