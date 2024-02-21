import openai  # Install if needed: pip install openai 
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain_openai import OpenAI
import os
sample_questions =[ 
    {
        'question1':'Thats impressive. In your previous role, you worked on an E-Commerce Project. Can you explain how you implemented the APIs to the front end in that project?',
        'Answer1':'i did not implemented any API'
    },
    {
        'question2':'Define HTML. How does front-end development employ it',
        'answer2':'Simply put, Hypertext Markup Language (HTML) is a markup language that is utilized for creating web pages. HTML describes the structure of a web page; it comprises a series of elements, such as headings, paragraphs, images, and links. Front-end developers use HTML to create the structure and content of a web page. They use HTML tags to define the different elements on a page and organize them into a logical hierarchy. Following this, the web browser interprets the HTML code to display the page. Without using HTML, the internet over the web as we know it today would not exist.'
    },
    {
        'question3':'Quesion: Define CSS. How is it applied to web page styling',
        'answer3':'HTML is the industry standard markup language for creating web pages. CSS, or Cascading Style Sheets, is a technique used for adding style to a web page. CSS is used to arrange the layout of a webpage, giving you control over aspects such as text color, font, spacing, text size, background pictures or colors, and much more. It styles HTML components rather than creating new ones. CSS may be embedded into HTML texts in three ways: inline, internal, and external, with external being the most commonly used method. With an external style sheet, you may change the whole appearance of the website by modifying only one file. CSS is used by front-end developers to design the layout of web pages, including changing the font, color, and size.'
    }
]

sample_skills = ['Frontend development'] 
def transcript_analysis(sample_skills):
    template = f"""
Analyze the following question-answer pair carefully. Prepare a Python dictionary for the following components.  Focus on the candidate's response and the relevance of their explanation to the given skills: {sample_skills}

 #'Average_Domain_similarity_between_questions_and_answers': Estimate how closely the answer aligns with the domain of the question (e.g.,  e-commerce, front-end development). Rate on a scale of 1-10 (1: irrelevant, 10: highly relevant),
#'Average_logical_similarity_between_Questions_answers': Assess if the answer logically addresses the question asked. Rate on a scale of 1-10 (1: illogical/unrelated, 10:  highly logical/directly answers),
#'interview_summary_score': Provide a general rating of the candidate's response. Consider clarity, conciseness, and technical correctness. Scale of 1-10 (1: poor, 10: excellent),
#'skill_score with respect to {sample_skills}': Rate the answer's demonstration of the skills listed. Does it show understanding or experience? Scale of 1-10 (1:  no demonstration, 10: strong expertise),
#'Explanation on the basis of your analysis': Provide a brief explanation for the scores assigned, highlighting strengths and weaknesses in the answer,
NOTE: Average = sum of the score given to each question pair and divided by the total number of question answer pairs
Provide your inputs below:

    'question': 'Thats impressive. In your previous role, you worked on an E-Commerce Project. Can you explain how you implemented the APIs to the front end in that project?',
    'answer': 'i did not implemented any API',
    'question2':'Define HTML. How does front-end development employ it',
    'answer2':'Simply put, Hypertext Markup Language (HTML) is a markup language that is utilized for creating web pages. HTML describes the structure of a web page; it comprises a series of elements, such as headings, paragraphs, images, and links. Front-end developers use HTML to create the structure and content of a web page. They use HTML tags to define the different elements on a page and organize them into a logical hierarchy. Following this, the web browser interprets the HTML code to display the page. Without using HTML, the internet over the web as we know it today would not exist.',
    'question3':'Quesion: Define CSS. How is it applied to web page styling',
    'answer3':'HTML is the industry standard markup language for creating web pages. CSS, or Cascading Style Sheets, is a technique used for adding style to a web page. CSS is used to arrange the layout of a webpage, giving you control over aspects such as text color, font, spacing, text size, background pictures or colors, and much more. It styles HTML components rather than creating new ones. CSS may be embedded into HTML texts in three ways: inline, internal, and external, with external being the most commonly used method. With an external style sheet, you may change the whole appearance of the website by modifying only one file. CSS is used by front-end developers to design the layout of web pages, including changing the font, color, and size.'
 
"""

    question1 = " On a scale of 1 to 10(wherever asked), calculate the components, when asked average, score the question answer pair individually and take the average of it"
    llm = llm = ChatOpenAI(temperature=0.1, openai_api_key=os.environ.get('OPENAPI_KEY'), model="gpt-3.5-turbo-0613")
    prompt  = PromptTemplate(template = template, input_variables = ["text1","text2"])
    conversation = LLMChain(llm = llm , prompt= prompt , verbose=True)
    with get_openai_callback() as cb:
        answer = conversation.run(question = question1)    
    return answer,cb

# def query_gpt35( text1, text2):
    
#     template= f"""
#     Carefully analyze the logical relationship between these two sentences. and answer the questions.

#     Sentence 1: {text1}
#     Sentence 2: {text2}

#     Generate a Python dictionary with the following rmat:'similar_topic_Score': "score", 'implication': implication_answer, 'explanation': explanation, 'explanation_score':"score",'Problem_solving':"numeric_score", 'Team_work':"numeric_score",'Python_skills':"numeric_score",'NLP':"numeric_score",'Machine learning Skills':"numeric_score",
#     use numerical representation for similarity.
#     """ 
    
#     question1 = " On a scale of 1 to 10, how semantically similar are these sentences ?"
#     llm = llm = ChatOpenAI(temperature=0, openai_api_key='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', model="gpt-3.5-turbo-0613")
#     prompt  = PromptTemplate(template = template, input_variables = ["text1","text2"])
#     conversation = LLMChain(llm = llm , prompt= prompt , verbose=True)
#     with get_openai_callback() as cb:
#         answer = conversation.run(question = question1)    
#     return answer,cb

# def question_compare(question,answer):
#     template = f"""
#     Carefully analyze the logical relationship between these the pair of given question and answer and answer the questions given to you.                
#     Sentence 1: {question}
#     Sentence 2: {answer}
#     Generate a Python dictionary with the following format: 'similar_topic_Score': "score", 'implication': implication_answer, 'explanation': explanation, 'explanation_score':"score"
#     use numerical representation for similarity.
    
#     """
#     question1 = " On a scale of 1 to 10, are these sentences imply the similar domain?"
#     llm = llm = ChatOpenAI(temperature=0, openai_api_key=venv.OPENAPI_KEY, model="gpt-3.5-turbo-0613")
#     prompt  = PromptTemplate(template = template, input_variables = ["text1","text2"])
#     conversation = LLMChain(llm = llm , prompt= prompt , verbose=True)
#     with get_openai_callback() as cb:
#         answer = conversation.run(question = question1)    
#     return answer,cb
    
# Example usage 
result,cb = transcript_analysis(sample_skills)
print(result,cb) 


#a new function to generate the score summary , interview summary , skill score , explanation of whole transcript
