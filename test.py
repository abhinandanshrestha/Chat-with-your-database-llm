import streamlit as st
from langchain_openai import OpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')
pwd=os.getenv('pass')
usr=os.getenv('user')
host=os.getenv('host')
dbase=os.getenv('dbase')

# Setup database
db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://{usr}:{pwd}@{host}:5432/{dbase}",
)

# setup llm
llm = OpenAI(temperature=0, openai_api_key=API_KEY)

# Create db chain using from_llm class method
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

# Streamlit UI layout
st.title("Database Query Application")

# Define function to prompt user for input
def get_prompt():
    question = st.text_input("Enter your question:")
    if question:
        try:
            st.write("Question:", question)
            sql_result = db_chain.run(question)
            st.write("SQL Result:", sql_result)
        except Exception as e:
            st.error(f"Error: {e}")

# Execute the prompt function
get_prompt()
