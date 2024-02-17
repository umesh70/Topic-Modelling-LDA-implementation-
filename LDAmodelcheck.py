import gensim
from gensim import corpora
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.matutils import hellinger
model = gensim.models.LdaModel.load('trained_model.gensim')
dictionary = corpora.Dictionary.load('build_dictionary.dict')

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

user_response = 'Machine Learning is a subset of artificial intelligence that focus on learning from data to develop an algorithm that can be used to make a prediction.Machine Learning uses a data-driven approach, It is typically trained on historical data and then used to make predictions on new data.ML can find patterns and insights in large datasets that might be difficult for humans to discover.'
llm_response = 'Machine learning is a subset of artificial intelligence (AI) that focuses on the development of algorithms and models that enable computers to learn from and make predictions or decisions based on data without being explicitly programmed. The core idea behind machine learning is to enable computers to automatically learn and improve from experience In traditional programming, a programmer writes rules and instructions for the computer to follow. However, in machine learning, the programmer provides the computer with data and algorithms that allow the computer to learn patterns and relationships within the data and make predictions or decisions based on that learning.'

buffer_responses = [user_response,llm_response]
cleaned_responses = text_preprocessing(buffer_responses)
lemmatizer  = WordNetLemmatizer()
filterdocs = [[lemmatizer.lemmatize(token) for token in doc] for doc in cleaned_responses]
finaldocs = [[token for token in doc if not token.isnumeric()]for doc in filterdocs]
cleaned_user = finaldocs[0]
cleaned_llm = finaldocs[1]

bow_user = dictionary.doc2bow(cleaned_user)
bow_llm = dictionary.doc2bow(cleaned_llm)

topics_user = model[bow_user]
topics_llm = model[bow_llm]
topics_userList = [prob for _,prob in topics_user]
topics_llmList = [prob for _,prob in topics_llm]

similarity =  1- hellinger(topics_userList,topics_llmList)
print("similarity:",similarity)