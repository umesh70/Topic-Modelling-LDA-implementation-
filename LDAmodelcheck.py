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
llm_response = 'A computer machine, commonly referred to simply as a computer, is an electronic device that is capable of performing a variety of tasks by executing sequences of instructions.Computers can process data in various forms, such as text, numbers, images, and sound. They can perform calculations, manipulate data, and generate outputs based on input.Computers have storage devices, such as hard disk drives (HDDs) or solid-state drives (SSDs), which allow them to store data and programs for later use.'

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