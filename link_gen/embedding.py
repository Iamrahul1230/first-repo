
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings




class Embedding:

  def __init__(self):
    pass

  # def get_embedding(self, sentences,user_input):
  #   model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
  #   embeddings = model.encode(sentences)

  #   return embeddings ,model   

  def user_inputs(self,user_input):
      # Create embeddings for the user question using a Google Generative AI model
      embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

      # Load a FAISS vector database from a local file
      new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

      # Perform similarity search in the vector database based on the user question
      docs = new_db.similarity_search(user_input,k=1, fetch_k=4,)

      print("docs result",docs)

      return docs
  
  
  def get_embedding(self,sentences,user_input):     
    # Create embeddings using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a vector store using FAISS from the provided text chunks and embeddings
    vector_store = FAISS.from_texts(sentences, embedding=embeddings)

    # Save the vector store locally with the name "faiss_index"
    vector_store.save_local("faiss_index")

    serached_result=self.user_inputs(user_input)[0].page_content

    return serached_result

  


  
