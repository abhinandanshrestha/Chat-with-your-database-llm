# Chat with your databases using LangChain

In this tutorial, we will be connecting to PostgreSQL database and initiating a conversation with it using Langchain without querying the database through SQL.

# **Getting Started**

## **Introduction to LangChain**

LangChain is an open-source library that offers developers a comprehensive set of resources to develop applications that run on Large Language Models (LLMs) by establishing a mechanism for linking LLMs to external data sources, such as personal documents or the internet. Developers can utilize LangChain to string together a sequence of commands to create sophisticated applications. In short, LangChain serves as a framework that enables the execution of a series of prompts to attain a specific outcome.

## **Why LangChain**

LangChain is an important tool for developers as it makes building of complex applications using LLMs easier. It allows users to connect LLMs to other data sources. The applications can act a wider range of information by connecting LLMs to other data sources. This makes the applications more powerful and versatile.

Langchain also provides features including:

- **Flexibility**: LangChain is a highly flexible and extensible framework that allows easy component swapping and chain customization to cater to unique requirements.
- **Speed**: The LangChain development team is continually enhancing the library’s speed, ensuring that users have access to the most recent LLM functionalities.
- **Community**: LangChain has a strong, engaged community where users can always seek assistance if necessary.

## **LangChain Structure**

The framework is organized into seven modules. Each module allows you to manage a different aspect of the interaction with the LLM.

![https://miro.medium.com/v2/resize:fit:700/1*z9LK7Yuahbb5U64rEbIDqg.jpeg](https://miro.medium.com/v2/resize:fit:700/1*z9LK7Yuahbb5U64rEbIDqg.jpeg)

Image credits: [LangChain 101: Build Your Own GPT-Powered Applications — KDnuggets](https://www.kdnuggets.com/2023/04/langchain-101-build-gptpowered-applications.html)

- **LLM:**LLM is the fundamental component of LangChain. It is a wrapper around the large language model which enables in utilization of the functionalities and capabilities of the model.
- **Chains:**Many a time, to solve tasks a single API call to an LLM is not enough. This module allows other tools to be integrated. For example, you may need to get data from a specific URL, summarize the returned text, and answer questions using the generated summary. This module allows multiple tools to be concatenated in order to solve complex tasks.
- **Prompts:** Prompts are at the core of any NLP application. It is how users interact with the model to try and obtain an output from it. It is important to to know how to write an effective prompt. LangChain provides prompt templates that enables users to format input and other utilities.
- **Document Loaders and Utils:** LangChain’s Document Loaders and Utils modules facilitate connecting to data sources and computations respectively. The utils module provides Bash and Python interpreter sessions amongst others. These are suitable for applications where the users need to interact directly with the underlying system or when code snippets are needed to compute a specific mathematical quantity or to solve a problem instead of computing answers at once.
- **Agents:** An agent is an LLM that makes a decision, takes an action and makes an observation on what has been done, and continues this cycle until the task is completed. LangChain library provides agents that can take actions based on inputs along the way instead of a hardcoded deterministic sequence.
- **Indexes:** The best models are often those that are combined with some of your textual data, in order to add context or explain something to the model. This module helps us do just that.
- **Memory:** This module enables users to create a persisting state between calls of a model. Being able to use a model that remembers what has been said in the past will improve our application. Short term → single chat history and Long term → multiple chat history

## **Applications of LangChain**

These are some of the applications of LangChain.

- **Querying Datasets with Natural Language**LLMs can write SQL queries using natural language. LangChain’s document loaders, index-related chains, and output parser help load and parse the data to generate results. Alternatively, inputting data structure to the LLM is a more common approach.
- **Interacting with APIs**LangChain’s chain and agent features enable users to include LLMs in a longer workflow with other API calls. This is useful for usecases, such as retrieving stock data or interacting with a cloud platform.
- **Building a Chatbot**Generative AI holds promise for chatbots that behave realistically. LangChain’s prompt templates provide control over the chatbot’s personality and responses. The message history tool allows for greater consistency within a conversation or even across multiple conversations by giving the chatbot a longer memory than what LLMs provide by default.

# **Creating a question answering app using LangChain**

In the last section we covered basic understanding of LangChain. In the following section we will build a question answering app using LangChain. Follow the steps given below to build a basic question answering app using LangChain.

## **Installing dependencies**

- Create and activate a virtual environment by executing the following command.

```
python -m venv myvenv
source myvenv/bin/activate #for ubuntu
myvenv/Scripts/activate #for windows
```

- Install libraries using pip.

```
pip install streamlit langchain_openai langchain python-dotenv psycopg2-binary

```

## **Setting up environment variables**

You can use any open source models with langchain. However openai models gives better results than the open source models. Openai key is required to access langchain if you are using any openai models. This tutorial designed with the openai model. Follow the steps to create a new openai key.

- Open [platform.openai.com](https://platform.openai.com/).
- Click on your name or icon option which is located on the top right corner of the page and select “API Keys” or click on the link — [Account API Keys — OpenAI API](https://platform.openai.com/account/api-keys).
- Click on **create new secret** key button to create a new openai key.

![https://miro.medium.com/v2/resize:fit:700/0*2lhYvsSRrJ34ER4k.png](https://miro.medium.com/v2/resize:fit:700/0*2lhYvsSRrJ34ER4k.png)

- Create a file named `.env` and add the openai key as follows.

```
OPENAI_API_KEY=<your_openai_key>
```

## **Creating simple LLM call using LangChain**

Create a new python file `langchain_demo.py` and add the following code to it.

```
from langchain.llms import OpenAI

# Accessing the OPENAI KEY
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

# Simple LLM call Using LangChain
llm = OpenAI(model_name="text-davinci-003", openai_api_key=API_KEY)
question = "Which language is used to create chatgpt ?"
print(question, llm(question))
```

We have imported the `OpenAI` wrapper from `langchain`. The OpenAI wrapper requires an openai key. The `OpenAI` key is accessed from the environment variables using the `environ` library. Initialize it to a `llm` variable with `text-davinci-003` model. Finally, define a question string and generate a response (`llm(question)`).

## **Run the script**

Run the LLM call using the following command.

```
python langchain_demo.py
```

You will get the output as follows.

![https://miro.medium.com/v2/resize:fit:700/1*UOnIAyMeDMGVVVRjccc_bw.png](https://miro.medium.com/v2/resize:fit:700/1*UOnIAyMeDMGVVVRjccc_bw.png)

## **Creating a prompt template**

Create a new python file `langchain_demo.py` and add the following code to it.

```
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

# Creating a prompt template and running the LLM chain
from langchain import PromptTemplate, LLMChain
template = "What are the top {n} resources to learn {language} programming?"
prompt = PromptTemplate(template=template,input_variables=['n','language'])
chain = LLMChain(llm=llm,prompt=prompt)
input = {'n':3,'language':'Python'}
print(chain.run(input))
```

We have imported `PromptTemplate` and `LLMChain` from `langchain`. Create a prompt template for getting top resources to learn a programming language by specifying `template` and the `input_variables`. Create a `LLMChain` and `chain.run()` method to run the LLM chain to get the result.

## **Run the script**

Run the LLM chain using the following command.

```
python langchain_demo.py
```

You will get the output as follows.

![https://miro.medium.com/v2/resize:fit:700/1*0V4Q_BGQj3vC57QaI_3ZYA.png](https://miro.medium.com/v2/resize:fit:700/1*0V4Q_BGQj3vC57QaI_3ZYA.png)

# **Interacting with databases using LangChain**

In this section, we will create an app to interact with the postgres database in a natural way (without querying it directly).

## **Installing postgres**

- Open the URL [Community DL Page (enterprisedb.com)](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads) and download the package based on your operating system. [[download for windows](https://sbp.enterprisedb.com/getfile.jsp?fileid=1258422)]
- Open the installer and continue with the default values. Specify a root password and complete the installation.

![https://miro.medium.com/v2/resize:fit:678/1*fSTFLpbPFG_ViycUiQqbug.png](https://miro.medium.com/v2/resize:fit:678/1*fSTFLpbPFG_ViycUiQqbug.png)

![https://miro.medium.com/v2/resize:fit:680/1*0Vie53v06_p4lws5Q--NmQ.png](https://miro.medium.com/v2/resize:fit:680/1*0Vie53v06_p4lws5Q--NmQ.png)

![https://miro.medium.com/v2/resize:fit:681/1*WOYjYQaw_x4emm8LkC6hDg.png](https://miro.medium.com/v2/resize:fit:681/1*WOYjYQaw_x4emm8LkC6hDg.png)

![https://miro.medium.com/v2/resize:fit:678/1*hR4WM9IWUFPU46ZPQSpj8w.png](https://miro.medium.com/v2/resize:fit:678/1*hR4WM9IWUFPU46ZPQSpj8w.png)

![https://miro.medium.com/v2/resize:fit:678/1*fKi1gH4vn-MUDROGA2tdsw.png](https://miro.medium.com/v2/resize:fit:678/1*fKi1gH4vn-MUDROGA2tdsw.png)

![https://miro.medium.com/v2/resize:fit:682/1*_pROnE1QonSUQ2EZBF2Awg.png](https://miro.medium.com/v2/resize:fit:682/1*_pROnE1QonSUQ2EZBF2Awg.png)

![https://miro.medium.com/v2/resize:fit:681/1*DAnJqR_16HZBTlgr7CbOPw.png](https://miro.medium.com/v2/resize:fit:681/1*DAnJqR_16HZBTlgr7CbOPw.png)

![https://miro.medium.com/v2/resize:fit:682/1*SUkM7HQmFaQgrPD8Y02JWA.png](https://miro.medium.com/v2/resize:fit:682/1*SUkM7HQmFaQgrPD8Y02JWA.png)

![https://miro.medium.com/v2/resize:fit:682/1*CMqPlY0IlmhEY06Om51b1A.png](https://miro.medium.com/v2/resize:fit:682/1*CMqPlY0IlmhEY06Om51b1A.png)

![https://miro.medium.com/v2/resize:fit:685/1*9bJ8LZ7vPbOotaiBo0AUiw.png](https://miro.medium.com/v2/resize:fit:685/1*9bJ8LZ7vPbOotaiBo0AUiw.png)

![https://miro.medium.com/v2/resize:fit:680/1*4YIVsApOSplafp09j72tmw.png](https://miro.medium.com/v2/resize:fit:680/1*4YIVsApOSplafp09j72tmw.png)

## **Creating database**

The postgres software has been installed. Create a database table called tasks to keep the task details. This database can be used as a data source for the langchain.

- Open pgAdmin4 application.
- Provide the root password to show the databases.

![https://miro.medium.com/v2/resize:fit:1920/1*VO22M4PsoLXTjDnZy6_RjA.png](https://miro.medium.com/v2/resize:fit:1920/1*VO22M4PsoLXTjDnZy6_RjA.png)

![https://miro.medium.com/v2/resize:fit:1920/1*J4b5S4OqNLLYCKzB0lQ3LQ.png](https://miro.medium.com/v2/resize:fit:1920/1*J4b5S4OqNLLYCKzB0lQ3LQ.png)

- Right click on the **databases** and select create → Database. Provide a database name and click on Save to finish the database creation.

![https://miro.medium.com/v2/resize:fit:1920/1*9Qi4MWbwKw-DMFFcQCcPrQ.png](https://miro.medium.com/v2/resize:fit:1920/1*9Qi4MWbwKw-DMFFcQCcPrQ.png)

![https://miro.medium.com/v2/resize:fit:868/1*YO4kj0uv1gZlSJRg6bdDwg.png](https://miro.medium.com/v2/resize:fit:868/1*YO4kj0uv1gZlSJRg6bdDwg.png)

## **Installing dependencies**

- Create and activate a virtual environment by executing the following command.

```
python -m venv venv
source venv/bin/activate #for ubuntu
venv/Scripts/activate #for windows
```

- Install `langchain`,`openai`, `python-environ` and `psycopg2` libraries using pip.

```
pip install streamlit langchain_openai langchain python-dotenv psycopg2-binary
```


## **Setup the SQL Database Chain**

Create a new python file `app.py` and add the following code to it.

```
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
```

**Understanding the code:**

- Import `langchain` modules `OpenAI`, `SQLDatabase`, and `SQLDatabaseChain`
- Access `OPENAI_API_KEY` from the environment variables file.
- Setup the database connection using `SQLDatabase.from_uri()` method by specifying the connection URL.
- Create `llm` object using `OpenAI()` by specifying the `temperature` and the `openai_api_key`.
- Create the database chain object called `db_chain` using `SQLDatabaseChain()` by specifying the `llm` and `database` objects.
- `get_prompt()` takes user input from the console and creates a query in the format by mentioning the question as an argument. It runs the SQL database chain using `db_chain.run()` method.

## **Runing the app**

Run the SQL database chain using the following command.

```
python app.py
```

You will get the output as follows,
![alt text](Screenshot.png)![alt text](Terminal.png)
There you have it! Your first langchain app in python :)