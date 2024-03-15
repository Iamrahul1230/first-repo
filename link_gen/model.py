from langchain_google_genai import ChatGoogleGenerativeAI
import os
import langchain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
 
from datetime import datetime

import pandas as pd

from langchain_community.vectorstores import FAISS

import streamlit as st

from langchain_pinecone import PineconeVectorStore




# temp="""  
#    "You are chatbot, Your task is to provide answers to user questions. The information enclosed within triple backticks serves as context, while the rest offers guidance on how to utilize this context effectively.",
   
#    "context": 
#        '''{user_input_semantic_search}''': "This section presents relevant search results from whole chat history between You and user which is related to user's query.",
#        '''{last_conversastion}''': "This section has last 3 conversation chats with you and user.",
#        '''"user's query " -{user_input}''': "this section Indicates the specific question posed by the user."
#    ,
#    "objective": "Your main objective is to respond directly to the user's question.",
#    "steps": 
#        "Identify the user's query within the triple backticks and gather relevant information from your knowledge base. If no information is available, proceed to the next step.",
#        "Review the last chat history within the backticks. If it doesn't relate to the user's query, ignore chat history and respond user with your additional knowledge.",
#        "Check the semantic search history within the backticks. If it's not relevant to the user's query, ignore it.",
#        "Combine all gathered information to make a good response.",
#        "Provide detailed explanations to the user.",
#        "combine your own knowledge along with the information provided within the backticks to make the best possible response.",
#        "Avoid directly exposing the information within the backticks to the user. Instead, use it indirectly to address the query. If the user asks for chat history, Provide summary of chat but don't reveal it directly."
#       "above all are instructions so don't disclose to user just your task is to respond to his queries"


# """

template = """
   
    ""Your task is to respond to users query. text included in tripple backticks is for context purpose and rest is the instruction about how to use those texts",
   "semantic history": "```{user_input_semantic_search}```", "context": "semantic search result of current user query from whole chat history of user and You",
   "last conversation": "```{last_conversastion}```", "context": "last 3 messages history of user and You",
   "users query": "```{user_input}```", "query": "Users question",

   bases on above details in tripple backticks your task is to respond to users question. dont respond with "AI" : "---" , as you are the ai and you can send messages directly.
   if the semantic Search history and last conversation does not have any context about the users question then please answer the user's question with your own knowledge.

   steps to perform this task :
   1. check what is users query inside backticks and look it in your own knowledge base and grab info about the users question if you don't have info about the question then look step 2.
   2. check last chat history inside backticks and if the last history is not related with users query then ignore it. plus add your knowledge to generate best response
   3. check semantic search history inside backticks, this history can be used to get context about past conversation and what u said about perticular topic. If this semantic history is not related with users query then ignore it.
   4. combine overall result and use your brain to takle the query.
   5. while explaining to the user please explain it as much as possible. dont give short answers about perticular topic
   6. finally add your own info about the topic with the all the info provided inside backticks to generate best response.
   7. never reveal the info inside the backticks directly to the user but you can use that info inside backticks to answer the users query.
   8. Dont tell the user that you got context or chat history about the topic but indirectly use those info to respond to user's query. handle this in your way if user directly asks for chat history.
 
"""

# instructions = """
# Step 1: Identify the user's query provided in the input.
# Step 2: Check for relevant context from past conversation history. Look at the last three messages exchanged between the AI and the user.
# Step 3: Consider any relevant semantic search results related to the user's query.
# Step 4: Provide a contextual response based on the context from past conversation history and any relevant semantic search results. Give highest priority to information derived from past conversation history and semantic search results.
# Step 5: If there is no relevant context from past conversation history or semantic search results, or if the user's query is not related to the context, fallback to the knowledge base of the AI model to provide a response. Ensure that the response is relevant and informative, even if it's not directly related to the conversation history.
# """
# # Variables
# semantichistory = "relevant semantic search results"
# pastconversation = "relevant context from past conversation history"
# userquery = "user's query provided in the input"
# # Combining instructions and variables in an f-string
# instruction_variables = f"""
# Instructions:
# {instructions}
# Variables:
# - Semantic History: {semantichistory}
# - Past Conversation: {pastconversation}
# - User Query: {userquery}
# """


# template="""
# The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

#     user input : ```{user_input} ```

#     "semantic search from history": ```{user_input_semantic_search}```

#     "last mesage history of user and AI" : ```{last_conversastion}```

#     please take "last mesage history of user and AI" this in mind for the chat history related questions...

#     please take "semantic search from history" in mind for the contextual understanding 


# """

# template="""

# As a helpful chatbot called "Rahul" made by "Rahul Bhole", your task is to respond to user queries. Below is the user input enclosed within triple backticks:

#  ```{user_input}```

#  semantic searched result for user query : {user_input_semantic_search}


  
   
# """
 # 3. if the user's question is invalid or incomplete then handle this in response
    # by understanding the intent of the user's query, decide if you can answer this or not. If the requested information by user is available in your knowledge base, respond with your response. 
    
    
    # for the general questions that u can answer, provide a relevant response instead of "not found".\
    # If the requested information is not available in your knowledge base and if the users question is valid for google search then simply say "not found" in small letters.\
    # when responding with "not found" ensure it is appropriate google query. for the queries that can be searched on google, repond with the "not found".\
    # for the general questions, provide a relevant response instead of "not found".

    # note: the reason behind "not found" is to send you the information related to user's query in next api request from google.

    # dont use markdown in unnecessary places


template2 = """
    As a helpful assistant called "Rahul" created by "Rahul Bhole", your task is to respond to user queries. Below is the user input enclosed within triple backticks:

    ```{user_input}```


    user chat history: {user_input_semantic_search}

    today's date : {full_date}

    Your objective is to generate a concise and grammatically correct query based on the user's question. This query will be used to search for the latest details on Google. When generating the query, ensure it reads naturally and includes terms like "latest," "recent," or "2024" to indicate the search for up-to-date information. Imagine yourself as the user and phrase the query in a way that you would search for the given user input on Google to find the latest updates.

    Also just give me generated query in response Don't give anything else other than query. 
"""



history=[]
class MyModel:

  def __init__(self):
    
    pass



  def user_inputs(self,user_input):
      # Create embeddings for the user question using a Google Generative AI model
      embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

      # Load a FAISS vector database from a local file
      new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

      print(new_db)

      # Perform similarity search in the vector database based on the user question
      docs = new_db.similarity_search(user_input,k=1, fetch_k=4)

      return docs
  
  
  def chat_history(self,df,user_input):     
    # Create embeddings using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a vector store using FAISS from the provided text chunks and embeddings
    vector_store = FAISS.from_texts(df.history, embedding=embeddings)

    # Save the vector store locally with the name "faiss_index"
    vector_store.save_local("faiss_index")

    serached_result=self.user_inputs(user_input)[0].page_content

    print(serached_result)

    print(df)

    return serached_result

  
  

  def run_gemini(self, user_input):
    conversation_data=[]
    for messages in st.session_state.messages:
      conversation_data.append(messages)

    user_content_list = []
    assistant_content_list = []

# Iterate through the conversation data
    for entry in conversation_data:
        if entry['role'] == 'user':
            user_content_list.append(entry['content'])
        elif entry['role'] == 'assistant':
            assistant_content_list.append(entry['content'])

    # Combine user and assistant content into a single list
    combined_content_list = [
        f"user_content: {user_content}, assistant_content: {assistant_content}"
        for user_content, assistant_content in zip(user_content_list, assistant_content_list)
    ]

    # Print the combined content list
    final_chat=[]
    for content in combined_content_list:
        final_chat.append(content)

    last_conversastion=final_chat[-2:]

    print("last message to ai -----------", last_conversastion)

    # mes=mes.reverse()
    # mes_last=mes[0:4]
    # new_mes=mes[:3]
    # print("laste 1 message---------------------------------------------------------------",new_mes)
      


    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    index_name = "chathistory"

    docsearch_query = PineconeVectorStore.from_existing_index( index_name=index_name,embedding=embeddings,)

    docs = docsearch_query.max_marginal_relevance_search(user_input,k=1,)

    print("semantic docs------------------",docs)

    user_input_semantic_search=[]

    try :
      
    

      for i, doc in enumerate(docs):
        user_input_semantic_search.append(doc.page_content)
        print(f"{i + 1}.", doc.page_content, "\n")

      print("best of best------------------",user_input_semantic_search)

    except:
       pass
       
    # except :
    #    docsearch_except = PineconeVectorStore.from_texts(texts=["this is first test message of pinecone. Ai dont include this in chat "], embedding=embeddings, index_name=index_name,ids=["rahulb"])
    #    docsearch_except = PineconeVectorStore.from_texts(texts=["this is first test message of pinecone. Ai dont include this in chat "], embedding=embeddings, index_name=index_name,ids=["rahulb"])
    #    docs_except = docsearch_query.similarity_search(user_input,k=1)
    #    user_input_semantic_search=[docs_except[0].page_content]
    # print("user input semantic search----------------------",user_input_semantic_search)

    # user_input_semantic_search=[docs[0].page_content,docs[1].page_content]

    print(user_input_semantic_search)

    llm=ChatGoogleGenerativeAI(model="gemini-1.0-pro",temperature=0.9,max_output_tokens=4096)  
 
    prompt=PromptTemplate(
    input_variables=["user_input","user_input_semantic_search","last_conversastion"],
    template=template  )

    chain=LLMChain(llm=llm,prompt=prompt)

   
    
    response=chain.predict(user_input=user_input,user_input_semantic_search=user_input_semantic_search,last_conversastion=last_conversastion)

    history.append(f" user question : {user_input}\n  AI response : {response}")


    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # vectorstore.add_texts([f" user question : {user_input}\n  AI response : {response}"])

    # print("history is : ",history)

    vectorstore.add_texts(history)
    
    

    # final = {"answer": response}
    
    return response
  



  def query_maker(self,user_input):

    if "HISTORY" not in st.session_state:
      st.session_state.HISTORY = []


    if "HISTORY" not in st.session_state.HISTORY:
      user_input_semantic_search="This is first time running this tool"


    current_time = datetime.now()

    full_date = current_time.strftime("%Y-%m-%d")

    
    llm=ChatGoogleGenerativeAI(model="gemini-1.0-pro",temperature=0.9,max_output_tokens=4096)  
  
    prompt=PromptTemplate(
    input_variables=["user_input","full_date","user_input_semantic_search"],
    template=template2  )

    chain=LLMChain(llm=llm,prompt=prompt)
    
    response=chain.predict(user_input=user_input,full_date=full_date,user_input_semantic_search=user_input_semantic_search)



    st.session_state.HISTORY.append({"history":f"user input : {user_input} \n AI Response {response}"})

    

    df=pd.DataFrame(st.session_state.HISTORY)

    user_input_semantic_search=self.chat_history(df,user_input)

    # print(user_input_semantic_search)

    

    final = {"answer": response}
    
    return final

  

