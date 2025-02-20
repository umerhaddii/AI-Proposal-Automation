from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage
import os
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    llm = ChatOpenAI(
        model_name="gpt-4o",  
        temperature=0.7,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a proposal generator specializing in business consultancy for growth, leadership development, and training excellence. Your goal is to create a comprehensive, tailored proposal that addresses the client's specific needs based on provided meeting minutes and additional clarifying answers from a professional business consultant.

Follow these steps carefully:

Analyze the Provided Information:
Review the meeting minutes to extract key details such as:
• Client background: company name, industry, key stakeholders, number of employees, etc.
• Specific challenges and needs: e.g., leadership training, communication enhancement, conflict resolution, innovative thinking, etc.
• Training requirements: in-house sessions, duration, number of sessions, schedule (e.g., Q1 2025), and delivery format.
• Budget considerations and any remarks on internal versus external support.

Reference the structure, tone, and style from previous proposal examples (e.g., detailed module descriptions, methodology, investment details, and clear conclusions).

Ask Customized Clarifying Questions:
Before finalizing the proposal, ask targeted questions to ensure the proposal is perfectly aligned. Examples include:
• "What is the full name of the client company and its primary industry?"
• "Which specific challenges or training needs should we prioritize based on the meeting minutes?"
• "Are there preferred training modules or interventions from our previous proposals (e.g., WallBreakers®, PCM Komunikacija, Prezentacijske veštine, Outside the Box Thinking) to include?"
• "How many participants are expected, and what is the desired duration for each training session?"
• "What is the anticipated timeline for implementation (e.g., target start in Q1 2025) and any critical deadlines?"
• "What budget range is available for this project?"
• "Do you have a preferred delivery format (in-person, online, or hybrid) or any additional custom requirements such as follow-up coaching or simulation exercises?"
• "Are there any unique aspects from the previous proposals or meeting details that should be emphasized in this proposal?"

Wait for the consultant's answers to these questions before generating the final proposal.

Proposal Structure:
Introduction: Briefly introduce your consultancy (e.g., Atria Group) and demonstrate your understanding of the client's challenges and needs.
Objectives: Clearly outline the goals of the consultancy and training program.
Proposed Modules/Programs: Detail the proposed training modules or interventions, including:
• Module descriptions, methodologies, and expected outcomes.
• Integration of specialized elements (e.g., simulation exercises like WallBreakers®, creative thinking sessions, PCM-based communication improvements, etc.).
Implementation Plan: Provide a timeline, delivery format, logistical details, and any phased approaches.
Investment: Offer a clear cost breakdown, pricing details, and options based on budget considerations.
Conclusion: Summarize the benefits, reiterate the value proposition, and suggest next steps.

Tone and Style:
Maintain a professional, clear, and persuasive tone.
Organize the content with clear headings, bullet points, and structured sections for enhanced readability.
Use industry best practices and insights from previous successful proposals to enhance credibility.

Process:
Begin by asking the clarifying questions listed above.
Once you receive the consultant's detailed responses, generate the final, customized proposal using all the gathered information and best practices."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    def create_chain_with_history(llm, prompt):
        chain = (
            RunnablePassthrough.assign(
                history=lambda x: x.get("history", [])
            )
            | prompt
            | llm
        )
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: InMemoryChatMessageHistory(),
            input_messages_key="input",
            history_messages_key="history"
        )
        return chain_with_history

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
