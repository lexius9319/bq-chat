import os
import re

import streamlit as st
from google.cloud import bigquery
import pandas as pd
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, AgentType
import json
import base64
from sqlalchemy import create_engine
from langchain_experimental.utilities import PythonREPL
from langchain.tools import Tool
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
)
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import AIMessage
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = st.secrets.OPENAI_API_KEY["key"]

# Initialize the BigQuery client
# client = bigquery.Client.from_service_account_info(st.secrets.BIGQUERY_CREDENTIALS_TOML)
cred_data = st.secrets.BIGQUERY_CREDENTIALS_TOML
json_str = json.dumps(cred_data.to_dict())

# Step 2: Encode the JSON string to bytes
json_bytes = json_str.encode('utf-8')

# Step 3: Encode the bytes to Base64
base64_bytes = base64.b64encode(json_bytes)

# Step 4: Convert the Base64-encoded bytes back to a string
base64_str = base64_bytes.decode('utf-8')


project = "kalibrr-analyze"
dataset = "kalibrr_analyze"

open_ai_model = "gpt-4o-mini"

sqlalchemy_url = f'bigquery://{project}/{dataset}?credentials_base64={base64_str}'
db = SQLDatabase.from_uri(sqlalchemy_url)

python_repl = PythonREPL()

def sql_agent_tools():
    tools = [
        Tool(
            name="python_repl",
            description=f"A Python shell. Use this to execute python commands. \
              Input should be a valid python command. \
              If you want to see the output of a value, \
              you should print it out with `print(...)`.",
            func=python_repl.run,
        ),
    ]
    return tools


# Example Queries
sql_examples = [
    {
        "input": "Count of Candidates by Country Code",
        "query": f"""
            SELECT
                country_code,
                COUNT(*) AS candidate_count
            FROM
                `{project}.{dataset}.candidates`
            GROUP BY
                country_code
            ORDER BY
                candidate_count DESC;
        """,
    },
    # {
    #     "input": "Average Age of Customers by Gender",
    #     "query": f"""
    #         SELECT
    #             gender,
    #             AVG(EXTRACT(YEAR FROM CURRENT_DATE()) - EXTRACT(YEAR FROM dob)) AS average_age
    #         FROM
    #             `{project}.{dataset}.customer`
    #         GROUP BY
    #             gender;
    #     """,
    # },
# ...
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    sql_examples,
    OpenAIEmbeddings(),
    FAISS,
    k=2,
    input_keys=["input"],
)

PREFIX = """
You are a SQL expert. You have access to a BigQuery database.
Identify which tables can be used to answer the user's question and write and execute a SQL query accordingly.
Given an input question, create a syntactically correct SQL query to run against the dataset kalibrr_analyze, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table; only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the information returned by these tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
If the question does not seem related to the database, just return "I don't know" as the answer.
If the user asks for a visualization of the results, use the python_agent tool to create and display the visualization.
After obtaining the results, you must use the mask_pii_data tool to mask the results before providing the final answer.
"""

SUFFIX = """Begin!
{chat_history}
Question: {input}
Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.
{agent_scratchpad}"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    prefix=PREFIX,
    suffix="",
    input_variables=["input", "top_k"],
    example_separator="\n\n",
)

messages = [
    SystemMessagePromptTemplate(prompt=few_shot_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    AIMessage(content=SUFFIX),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]

prompt = ChatPromptTemplate.from_messages(messages)
extra_tools = sql_agent_tools()

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, input_key="input"
)

model = ChatOpenAI(model="gpt-4o", temperature=0)

# Create the agent executor
agent_executor = create_sql_agent(
    llm=model,
    db=db,
    verbose=True,
    top_k=10,
    prompt=prompt,
    # extra_tools=extra_tools,
    input_variables=["input", "agent_scratchpad", "chat_history"],
    agent_type="openai-tools",
    agent_executor_kwargs={"handle_parsing_errors": True, "memory": memory},
)
st.title("Data Analysis Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask your question:")

if st.button("Run Query"):
    if user_input:
        with st.spinner("Processing..."):
            st.session_state.history.append(f"User: {user_input}")
            response = agent_executor.run(input=user_input)
            if "sandbox:" in response:
                response = response.replace(f"sandbox:", "")
            match = re.search(r"\((.+\.png)\)", response)
            if match:
                image_file_path = match.group(1)
                if os.path.isfile(image_file_path):
                    st.session_state.history.append({"image": image_file_path})
                else:
                    st.error("The specified image file does not exist.")
            else:
                st.session_state.history.append(f"Agent: {response}")
            st.rerun()
    else:
        st.error("Please enter a question.")

for message in st.session_state.history:
    if isinstance(message, str):
        st.write(message)
    elif isinstance(message, dict) and "image" in message:
        st.image(message["image"])
