from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def retrieving_data_from_db():
    text_field = "text"
    index_name = "wilson-embeddings"
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


def create_chain(retriever):
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a professional wilsons disease consultant and you have to 
                    answer the user's questions based on the context: {context}.
                    If user greets or asks about you, tell about yourself, your expertise and how you can assist the user. 
                    Restrict yourself from generating response if asked anything out of the context info. provided.
                    Highlight the important points if necessary and in plain and understandable english.
                    Also give the source of information for the answers you give at the end in a seperate line.
                    Make sure you generate the response in paragraphs of 3-4 lines.
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # chain = prompt | model
    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "human",
                "Given the above conversation,generate a search query to look up to the information relevent to the conversation.",
            ),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=model, retriever=retriever, prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    return retrieval_chain


def process_chat(chain, user_input, chat_history):

    response = chain.invoke({"input": user_input, "chat_history": chat_history})

    return response["answer"]


# retriever = retrieving_data_from_db()
# print("Entering chain")
# chain = create_chain(retriever)
# chat_history = []
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         break
#     response = process_chat(chain, user_input, chat_history)
#     chat_history.append(HumanMessage(content=user_input))
#     chat_history.append(AIMessage(content=response))
#     # print(chat_history)
#     print("Assistant: ", response)
#     print("\n")
