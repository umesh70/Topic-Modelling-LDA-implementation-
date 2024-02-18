import openai  # Install if needed: pip install openai 
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain_openai import OpenAI
import venv

# Set your OpenAI API key
#openai.api_key = "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" 

# Sample texts
text1 = "In-depth research and tailored communication are key. Could you elaborate on your background in photography and how it relates to your sales career?"
text2 = " I dont have a background in photography." 

def query_gpt35( text1, text2):
    
    template= f"""
    Carefully analyze the logical relationship between these two sentences. and answer the questions.

    Sentence 1: {text1}
    Sentence 2: {text2}

    Generate a Python dictionary with the following rmat:'similar_topic_Score': "score", 'implication': implication_answer, 'explanation': explanation, 'explanation_score':"score",'Problem_solving':"numeric_score", 'Team_work':"numeric_score",'Python_skills':"numeric_score",'NLP':"numeric_score",'Machine learning Skills':"numeric_score",
    use numerical representation for similarity.
    """ 
    
    question1 = " On a scale of 1 to 10, how semantically similar are these sentences ?"
    llm = llm = ChatOpenAI(temperature=0, openai_api_key='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', model="gpt-3.5-turbo-0613")
    prompt  = PromptTemplate(template = template, input_variables = ["text1","text2"])
    conversation = LLMChain(llm = llm , prompt= prompt , verbose=True)
    with get_openai_callback() as cb:
        answer = conversation.run(question = question1)    
    return answer,cb

def question_compare(question,answer):
    template = f"""
    Carefully analyze the logical relationship between these the pair of given question and answer and answer the questions given to you.                
    Sentence 1: {question}
    Sentence 2: {answer}
    Generate a Python dictionary with the following format: 'similar_topic_Score': "score", 'implication': implication_answer, 'explanation': explanation, 'explanation_score':"score"
    use numerical representation for similarity.
    
    """
    question1 = " On a scale of 1 to 10, are these sentences imply the similar domain?"
    llm = llm = ChatOpenAI(temperature=0, openai_api_key=venv.OPENAPI_KEY, model="gpt-3.5-turbo-0613")
    prompt  = PromptTemplate(template = template, input_variables = ["text1","text2"])
    conversation = LLMChain(llm = llm , prompt= prompt , verbose=True)
    with get_openai_callback() as cb:
        answer = conversation.run(question = question1)    
    return answer,cb
    
# Example usage 
result,cb = query_gpt35(text1,text2)
print(result,cb) 
