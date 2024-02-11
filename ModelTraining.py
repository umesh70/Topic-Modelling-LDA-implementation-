import nltk
import PyPDF2
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pprint import pprint
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def reading_data(pdf_path):
    with open(pdf_path,'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        
        text= ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text+=page_text
    return text

doc1_path  = "2402.05679.pdf"
doc2_path = "2201.01943.pdf"
doc3_path = "0520.pdf"
doc1_text = reading_data(doc1_path)
doc2_text = reading_data(doc2_path)
doc3_text = reading_data(doc3_path)

docs = [doc1_text,doc2_text,doc3_text]

def text_preprocessing(docs):
    #tokenzation
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()
        docs[idx] = tokenizer.tokenize(docs[idx])

        
    #remove numbers
    docs = [[token for token in doc if not token.isnumeric()]for doc in docs]
    #remove words with one character
    docs = [[token for token in doc if len(token)>1] for doc in docs]
    #remove stop words
    english_words = stopwords.words('english')
    docs = [[token for token in doc if token not in english_words] for doc in docs]
    return docs

cleaned_docs = text_preprocessing(docs)
lemmatizer  = WordNetLemmatizer()
filterdocs = [[lemmatizer.lemmatize(token) for token in doc] for doc in cleaned_docs]
#remove numbers
finaldocs = [[token for token in doc if not token.isnumeric()]for doc in filterdocs]
dictionary= Dictionary(finaldocs)
corpus = [dictionary.doc2bow(doc) for doc in finaldocs]
print('number of unique tokens:%d' % len(dictionary))
print('number of documents: %d' % len(corpus))

num_of_topics = 3
chunksize = 1
passes = 30
iteration = 100
eval_every = None

temp = dictionary[0]
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iteration,
    num_topics=num_of_topics,
    passes=passes,
    eval_every=eval_every
)
model.save('trained_model.gensim')
dictionary.save('build_dictionary.dict')