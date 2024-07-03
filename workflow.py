from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing import Dict, List
from config import LLAMA_API
from data_preparation import query_db
from config import LLAMA_API, LLAMA_MODEL

model = ChatOpenAI(
    model_name=LLAMA_MODEL,
    openai_api_key=LLAMA_API,
    openai_api_base="https://api.llama-api.com"
)

DB_DESCRIPTION = """You have access to the following tables and columns in a SQLite3 database:

Retail Table
Customer_ID: A unique ID that identifies each customer.
Name: The customer's name.
Gender: The customer's gender: Male, Female.
Age: The customer's age.
Country: The country where the customer resides.
State: The state where the customer resides.
City: The city where the customer resides.
Zip_Code: The zip code where the customer resides.
Product: The product purchased by the customer.
Category: The category of the product.
Price: The price of the product.
Purchase_Date: The date when the purchase was made.
Quantity: The quantity of the product purchased.
Total_Spent: The total amount spent by the customer.
"""

class CanAnswerOutput(BaseModel):
    reasoning: str = Field(description="The reasoning behind whether the question can be answered")
    can_answer: bool = Field(description="Whether the question can be answered or not")

can_answer_parser = PydanticOutputParser(pydantic_object=CanAnswerOutput)

can_answer_router_prompt = PromptTemplate(
    template="""system
    You are a database reading bot that can answer users' questions using information from a database. \n

    {data_description} \n\n

    Given the user's question, decide whether the question can be answered using the information in the database. \n\n

    Return a JSON with two keys, 'reasoning' and 'can_answer', and no preamble or explanation.
    Return one of the following JSON:

    {{"reasoning": "I can find the average total spent by customers in California by averaging the Total_Spent column in the Retail table filtered by State = 'CA'", "can_answer":true}}
    {{"reasoning": "I can find the total quantity of products sold in the Electronics category using the Quantity column in the Retail table filtered by Category = 'Electronics'", "can_answer":true}}
    {{"reasoning": "I can't answer how many customers purchased products last year because the Retail table doesn't contain a year column", "can_answer":false}}

    user
    Question: {question} \n
    assistant""",
    input_variables=["data_description", "question"],
)

can_answer_router = can_answer_router_prompt | model | can_answer_parser

def check_if_can_answer_question(state):
    result = can_answer_router.invoke({"question": state["question"], "data_description": DB_DESCRIPTION})
    return {"plan": result.reasoning, "can_answer": result.can_answer}

def skip_question(state):
    if state["can_answer"]:
        return "no"
    else:
        return "yes"

write_query_prompt = PromptTemplate(
    template="""system
    You are a database reading bot that can answer users' questions using information from a database. \n

    {data_description} \n\n

    In the previous step, you have prepared the following plan: {plan}

    Return an SQL query with no preamble or explanation. Don't include any markdown characters or quotation marks around the query.
    user
    Question: {question} \n
    assistant""",
    input_variables=["data_description", "question", "plan"],
)

write_query_chain = write_query_prompt | model | StrOutputParser()

def write_query(state):
    result = write_query_chain.invoke({
        "data_description": DB_DESCRIPTION,
        "question": state["question"],
        "plan": state["plan"]
    })
    return {"sql_query": result}

def execute_query(state):
    query = state["sql_query"]
    try:
        return {"sql_result": query_db(query).to_markdown()}
    except Exception as e:
        return {"sql_result": str(e)}

write_answer_prompt = PromptTemplate(
    template="""system
    You are a database reading bot that can answer users' questions using information from a database. \n

    In the previous step, you have planned the query as follows: {plan},
    generated the query {sql_query}
    and retrieved the following data:
    {sql_result}

    Return a text answering the user's question using the provided data.
    user
    Question: {question} \n
    assistant""",
    input_variables=["question", "plan", "sql_query", "sql_result"],
)

write_answer_chain = write_answer_prompt | model | StrOutputParser()

def write_answer(state):
    result = write_answer_chain.invoke({
        "question": state["question"],
        "plan": state["plan"],
        "sql_result": state["sql_result"],
        "sql_query": state["sql_query"]
    })
    return {"answer": result}

cannot_answer_prompt = PromptTemplate(
    template="""system
    You are a database reading bot that can answer users' questions using information from a database. \n

    You cannot answer the user's questions because of the following problem: {problem}.

    Explain the issue to the user and apologize for the inconvenience.
    user
    Question: {question} \n
    assistant""",
    input_variables=["question", "problem"],
)

cannot_answer_chain = cannot_answer_prompt | model | StrOutputParser()

def explain_no_answer(state):
    result = cannot_answer_chain.invoke({
        "problem": state["plan"],
        "question": state["question"]
    })
    return {"answer": result}

class WorkflowState(Dict):
    question: str
    plan: str
    can_answer: bool
    sql_query: str
    sql_result: str
    answer: str

workflow = StateGraph(WorkflowState)

workflow.add_node("check_if_can_answer_question", check_if_can_answer_question)
workflow.add_node("write_query", write_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("write_answer", write_answer)
workflow.add_node("explain_no_answer", explain_no_answer)

workflow.set_entry_point("check_if_can_answer_question")

workflow.add_conditional_edges(
    "check_if_can_answer_question",
    skip_question,
    {
        "yes": "explain_no_answer",
        "no": "write_query",
    },
)

workflow.add_edge("write_query", "execute_query")
workflow.add_edge("execute_query", "write_answer")

workflow.add_edge("explain_no_answer", END)
workflow.add_edge("write_answer", END)

app = workflow.compile()