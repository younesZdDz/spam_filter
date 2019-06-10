import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from html import unescape
import urlextract 
from nltk.stem import WordNetLemmatizer 



class TextProcessor:
    """
    Class for carrying all the text pre-processing stuff throughout the project
    """

    def __init__(self):
        
        self.stopwords = stopwords.words('english')

        #self.ps = PorterStemmer()  
        self.lm = WordNetLemmatizer()
        # stemmer will be used for each unique word once
        #self.stemmed = dict()
        self.lemmetized = dict()

        self.url_extractor = urlextract.URLExtract()
        

    
    def process(self, text, allow_stopwords = False, use_stemmer = True) :
        """
        Process the specified text,
        splitting by non-alphabetic symbols, casting to lower case,
        removing stopwords, HTML tags and stemming each word

        :param text: text to precess
        :param allow_stopwords: whether to remove stopwords
        :return: processed text
        """
        ret = []

        # split and cast to lower case
        #text = re.sub(r'<[^>]+>', ' ', str(text))        
        text = text.lower()
        text = re.sub(r'[0-9]+(?:\.[0-9]+){3}', ' URL ', text)
        urls = list(set(self.url_extractor.find_urls(text)))
        urls.sort(key=lambda url: len(url), reverse=True)
        for url in urls:
            text = text.replace(url, " URL ")
            
        text = re.sub('<head.*?>.*?</head>', '', text, flags=re.M | re.S | re.I)
        text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
        text = re.sub('<.*?>', '', text, flags=re.M | re.S)
        text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
        text = unescape(text)
        text = re.sub(r'\W+', ' ', text, flags=re.M)
       
        
        text= re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)    
        
        for word in text.split():
            # remove non-alphabetic and stop words
            if (word.isalpha() and word not in self.stopwords) or allow_stopwords:
                if use_stemmer:
                    if word not in self.lemmetized:
                        self.lemmetized[word] = self.lm.lemmatize(word)
                    # use stemmed version of word
                    ret.append(self.lemmetized[word])
                else: 
                    ret.append(word)
        return ' '.join(ret)