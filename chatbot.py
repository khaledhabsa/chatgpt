import os
import openai
import logging
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler


from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from langchain.document_loaders import (UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredFileLoader)
import langchain.text_splitter as text_splitter
from langchain.text_splitter import (RecursiveCharacterTextSplitter, CharacterTextSplitter)

from typing import List
import streamlit


REQUEST_TIMEOUT = 50



logging.basicConfig(level=logging.INFO)




class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class DocChatbot:
    # Function to expand the user's query based on the provided context and index_topic
        def expand_user_query(self, user_question, index_topic):
            expanded_prompt = f"""We have a knowledge base in a Vector Index database of some topic, 
            and we get some brief questions from our users about the content of this index. 
            We need to expand the questions of the users in a way that helps us sense their actual intent 
            of the question and make a better search in the Vector database to fulfill their need. 
            I am providing you with these contextual information: Knowledge base topic: {index_topic}, 
            Users Original brief question: {user_question}. 
            I want you to: 
            - Firstly analyze the user's original brief question and provide a brief summary of it, getting the Key Concepts from it,
            - then explain the meaning or context of each key concept from the user's query
            - then based on the previous step's result try to Explore potential connections between the concepts from the Knowledge base topic and the explained key concepts from the user's query."""
            response = self.llm.create_completion(
                prompt=f'{user_question} & {index_topic} + {expanded_prompt}',
                max_tokens=100
            )
            expanded_query = response['choices'][0]['text'].strip()
            return expanded_query

    
        # Function to generate the final answer based on the user's query and the enhanced similarity context
        def generate_final_answer(self, user_question, enhanced_similarity):
            full_prompt = f'{user_question} & Contextual information: {enhanced_similarity}'
            remaining_length = 2000 - len(full_prompt)
            response = self.llm.create_completion(
                prompt=full_prompt,
                max_tokens=remaining_length
            )
            final_answer = response['choices'][0]['text'].strip()
            return final_answer


class DocChatbot:
    llm: ChatOpenAI
    condens_question_llm: ChatOpenAI
    embeddings: OpenAIEmbeddings
    vector_db: FAISS
    chatchain: BaseConversationalRetrievalChain

    def __init__(self) -> None:
        #init for LLM and Embeddings
        load_dotenv()
        assert(os.getenv("OPENAI_API_KEY") is not None)
        api_key = str(os.getenv("OPENAI_API_KEY"))
        embedding_deployment = "text-embedding-ada-002"

        #check if user is using API from openai.com or Azure OpenAI Service by inspecting the api key
        if api_key.startswith("sk-"):
            # user is using API from openai.com
            assert(len(api_key) == 51)

            self.llm = ChatOpenAI(
                temperature=0,
                openai_api_key=api_key,
                request_timeout=REQUEST_TIMEOUT,
            ) # type: ignore
        else:
            # user is using Azure OpenAI Service
            assert(os.getenv("OPENAI_GPT_DEPLOYMENT_NAME") is not None)
            assert(os.getenv("OPENAI_API_BASE") is not None)
            assert(len(api_key) == 32)

            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("OPENAI_GPT_DEPLOYMENT_NAME"),
                temperature=0,
                openai_api_version="2023-05-15",
                openai_api_type="azure",
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                openai_api_key=api_key,
                request_timeout=REQUEST_TIMEOUT,
            ) # type: ignore

            embedding_deployment = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")

        self.condens_question_llm = self.llm

        self.embeddings = OpenAIEmbeddings(
            deployment=embedding_deployment, 
            chunk_size=1
            ) # type: ignore

    def init_streaming(self, condense_question_container, answer_container) -> None:
        api_key = str(os.getenv("OPENAI_API_KEY"))
        if api_key.startswith("sk-"):
            # user is using API from openai.com
            self.llm = ChatOpenAI(
                temperature=0,
                openai_api_key=api_key,
                request_timeout=REQUEST_TIMEOUT,
                streaming=True,
                callbacks=[StreamHandler(answer_container)]
            ) # type: ignore

            self.condens_question_llm = ChatOpenAI(
                temperature=0,
                openai_api_key=api_key,
                request_timeout=REQUEST_TIMEOUT,
                streaming=True,
                callbacks=[StreamHandler(condense_question_container, "🤔...")]
            ) # type: ignore
        else:
            # user is using Azure OpenAI Service
            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("OPENAI_GPT_DEPLOYMENT_NAME"),
                temperature=0,
                openai_api_version="2023-05-15",
                openai_api_type="azure",
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                request_timeout=REQUEST_TIMEOUT,
                streaming=True,
                callbacks=[StreamHandler(answer_container)]
            ) # type: ignore

            self.condens_question_llm = AzureChatOpenAI(
                deployment_name=os.getenv("OPENAI_GPT_DEPLOYMENT_NAME"),
                temperature=0,
                openai_api_version="2023-05-15",
                openai_api_type="azure",
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                request_timeout=REQUEST_TIMEOUT,
                streaming=True,
                callbacks=[StreamHandler(condense_question_container, "🤔...")]
            ) # type: ignore
        



    def init_chatchain(self, chain_type : str = "stuff") -> None:
        # init for ConversationalRetrievalChain
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up input, rephrase the standalone question. 
        The standanlone question to be generated should be in the same language with the input. 
        For example, if the input is in chinees, the follow up question or the standalone question below should be in the same language.
        Only Answer the question based on the context below. if the question cannot be answerd using the information provided answer with "I don't know" don't try to make up an answer.
            
        Chat History:
            {chat_history}

        Follow Up Input:
            {question}


        Standalone Question: """
            )   


        #CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""I want you to act as a Whatsapp customer support to help the customers in any question and the all processes should be completed through WhatsApp. 
        #Given the following conversation and a follow up input, rephrase the standalone question. 
        #The standanlone question to be generated should be in the same language with the input. 
        #For example, if the input is in chinees, the follow up question or the standalone question below should be in the same language.
        
        #    Chat History:
        #    {chat_history}

        #    Follow Up Input:
        #    {question}

        #    Standalone Question:"""
        #    )   

        # stuff chain_type seems working better than others
        self.chatchain = ConversationalRetrievalChain.from_llm(llm=self.llm, 
                                                retriever=self.vector_db.as_retriever(),
                                                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                                condense_question_llm=self.condens_question_llm,
                                                chain_type=chain_type,
                                                return_source_documents=True,
                                                verbose=False)
                                                # combine_docs_chain_kwargs=dict(return_map_steps=False))

    # get answer from query, return answer and source documents
    def get_answer_with_source(self, query, chat_history):
      try:
          result = self.chatchain({
                  "question": query,
                  "chat_history": chat_history
          },
          return_only_outputs=True)
          
          return result['answer'], result['source_documents']
      except openai.error.InvalidRequestError as e:
        if e.error.code == "content_filter" and e.error.innererror:
            content_filter_result = e.error.innererror.content_filter_result
            # print the formatted JSON
            print(content_filter_result)

            # or access the individual categories and details
            for category, details in content_filter_result.items():
                print(f"{category}:\n filtered={details['filtered']}\n severity={details['severity']}")
        else:
            print("An error occurred: " + str(e))
            return None, None

    # get answer from query. 
    # This function is for streamlit app and the chat history is in a format aligned with openai api
    def get_answer(self, query, chat_history):
        ''' 
        Here's the format for chat history:
        [{"role": "assistant", "content": "How can I help you?"}, {"role": "user", "content": "What is your name?"}]
        The input for the Chain is in a format like this:
        [("How can I help you?", "What is your name?")]
        That is, it's a list of question and answer pairs.
        So need to transform the chat history to the format for the Chain
        '''  
        chat_history_for_chain = []

        for i in range(0, len(chat_history), 2):
            chat_history_for_chain.append((
                chat_history[i]['content'], 
                chat_history[i+1]['content'] if chat_history[i+1] is not None else ""
                ))
        try:
            result = self.chatchain({
                    "question": query,
                    "chat_history": chat_history_for_chain
            },
            return_only_outputs=True)
            
            return result['answer'], result['source_documents']
        except openai.error.InvalidRequestError as e:
            if e.error.code == "content_filter" and e.error.innererror:
                content_filter_result = e.error.innererror.content_filter_result
                # print the formatted JSON
                print(content_filter_result)

                # or access the individual categories and details
                for category, details in content_filter_result.items():
                    print(f"{category}:\n filtered={details['filtered']}\n severity={details['severity']}")
            else:
                print("An error occurred: " + str(e))
                return None, None
        

    # load vector db from local
    def load_vector_db_from_local(self, path: str, index_name: str):
        self.vector_db = FAISS.load_local(path, self.embeddings, index_name)
        print(f"Loaded vector db from local: {path}/{index_name}")

    # save vector db to local
    def save_vector_db_to_local(self, path: str, index_name: str):
        FAISS.save_local(self.vector_db, path, index_name)
        print("Vector db saved to local")


    # split documents, generate embeddings and ingest to vector db
    def init_vector_db_from_documents(self, file_list: List[str]):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        docs = []
        for file in file_list:
            print(f"Loading file: {file}")
            ext_name = os.path.splitext(file)[-1]
            # print(ext_name)

            if ext_name == ".pptx":
                loader = UnstructuredPowerPointLoader(file)
            elif ext_name == ".docx":
                loader = UnstructuredWordDocumentLoader(file)
            elif ext_name == ".pdf":
                print("it's pdf")
                loader = PyPDFLoader(file)
            else:
                # process .txt, .html
                loader = UnstructuredFileLoader(file)

            doc = loader.load_and_split(text_splitter)  
            print(doc[:60])          
            docs.extend(doc)
            print("Processed document: " + file)
    
        print("Generating embeddings and ingesting to vector db.")

        self.vector_db = FAISS.from_documents(docs, self.embeddings)
        print("Vector db initialized.")

        
