from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage
import os
import logging
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Initialize the language model using Streamlit secrets
    llm = ChatOpenAI(
        model_name="gpt-4o",  
        temperature=0.7,
        api_key=st.secrets["OPENAI_API_KEY"]  # Use Streamlit secrets
    )

    # Create the conversation prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a proposal generator..."""),  # Your existing prompt
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    def create_chain_with_history(llm, prompt):
        # Create a runnable that can handle history
        chain = (
            RunnablePassthrough.assign(
                history=lambda x: x.get("history", [])
            )
            | prompt
            | llm
        )

        # Create a wrapper with message history
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: InMemoryChatMessageHistory(),
            input_messages_key="input",
            history_messages_key="history"
        )

        return chain_with_history

    # Initialize the chain with history
    chain_with_history = create_chain_with_history(llm, prompt)

    def get_bot_response(user_input):
        try:
            response = chain_with_history.invoke(
                {"input": user_input},
                {"configurable": {"session_id": "default_session"}}
            )
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Please try again or rephrase your question."

except Exception as e:
    logger.error(f"Error initializing the application: {str(e)}")
    def get_bot_response(user_input):
        return "System is currently unavailable. Please try again later."