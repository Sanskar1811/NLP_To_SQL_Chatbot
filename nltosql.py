import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama
import uvicorn

# Initialize GPT-based LLM via Ollama
llm = ChatOllama(temperature=0, model="llama3.2")

# Define the prompt template
prompt_template = """
You are an intelligent assistant that helps convert natural language queries into SQL queries. 
Your task is to take a natural language question and return only the SQL query. Do not provide any explanations, just return the query.

IMPORTANT:
- Only return the SQL query.
- Do not include any text, explanations, or commentary before or after the SQL query.
- Ensure the SQL query is clean and without any additional symbols, text, or commentary.

The table you are working with is named `staff_data`. Here are the details of the table and its columns:

Table Name: staff_data

1. id: Integer – Unique row identifier.
2. bds_id: String – Internal BDS ID.
3. current_project_id: String – Project ID assigned.
4. project_name: String – Name of the project.
5. btb: String – Type (e.g., BTB, STZ, B2B).
6. title: String – Title such as Mr, Ms, Dr.
7. first_name: String – First name of staff.
8. middle_initial: String – Middle initial.
9. last_name: String – Last name.
10. job_title: String – Job title of the person.
11. job_function: String – Function/department.
12. job_level: String – Level in the organization.
13. connections: Integer – Number of connections.
14. work_experience: Integer – Years of experience.
15. staff_url_linkedin_url: String – LinkedIn profile link.
16. staff_url: String – Staff profile link on company site.
17. company: String – Name of the company.
18. address1: String – Primary address line.
19. address2: String – Secondary address line.
20. address3: String – Tertiary address line.
21. city: String – City of work.
22. state: String – State.
23. postalcode: String – Postal code.
24. country: String – Country.
25. region: String – Business region.
26. email: String – Confirmed email.
27. assumed_email: String – Assumed email.
28. email_url: String – Source of email info.
29. workphone_direct_tel: Integer – Direct telephone number.
30. otherphone_company_tel: Integer – Company telephone number.
31. mobile: Integer – Mobile phone number.
32. website: String – Company website.
33. web_status: String – Web status (Active/UnActive).
34. web_remarks: String – Additional website info.
35. voice_status: String – Voice verification status.
36. voice_remarks: String – Voice verification notes.
37. industry: String – Industry sector.
38. market: String – Market region (e.g., US Market, Indian Market).
39. sic_code: String – SIC industry classification code.
40. revenue: String – Revenue range.
41. employee_size: String – Employee size range.
42. uploaded_status: String – Upload status.
43. merger_status: String – Whether merged or not.
44. delivery_date: Date – Scheduled delivery.
45. imported_by: String – User who imported the data.
46. updated_by: String – User who last updated the data.
47. updated_at: Date – Last update timestamp.
48. created_at: Date – Creation timestamp.

Now, based on the information provided, convert the following natural language question into an SQL query.

Ensure that you return only the SQL query, with no additional explanations, commentary, or noise.

Question: {question}
"""

# Define the prompt template using Langchain's PromptTemplate class
prompt = PromptTemplate(
    input_variables=["question"],
    template=prompt_template
)

# Create a Langchain pipeline that takes a natural language query and converts it to SQL
nl_to_sql_chain = LLMChain(prompt=prompt, llm=llm)

# FastAPI app initialization
app = FastAPI()

# Pydantic model for the input
class QueryRequest(BaseModel):
    question: str

# SQLite connection setup
db_conn = sqlite3.connect('sample_contacts_dataset.db')

# Function to execute SQL and return results in a Pandas DataFrame
def execute_sql_and_return_df(sql_query):
    try:
        result = pd.read_sql_query(sql_query, db_conn)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error executing SQL query: {e}")

# Define FastAPI route to generate SQL from user query
@app.post("/generate_sql/")
async def generate_sql(request: QueryRequest):
    # Generate SQL from natural language input
    response = nl_to_sql_chain.invoke({"question": request.question})
    sql_query = response.get("text", "").strip()  # Access the 'text' field and strip whitespace
    
    if not sql_query:
        raise HTTPException(status_code=400, detail="Generated SQL query is empty.")

    # Return the generated SQL query
    return {"sql_query": sql_query}

# Define FastAPI route to execute SQL and return results
@app.post("/execute_sql/")
async def execute_sql(request: QueryRequest):
    # Generate SQL from the user's natural language question
    response = nl_to_sql_chain.invoke({"question": request.question})
    sql_query = response.get("text", "").strip()

    if not sql_query:
        raise HTTPException(status_code=400, detail="Generated SQL query is empty.")

    # Execute SQL and return results
    result_df = execute_sql_and_return_df(sql_query)
    return {"result": result_df.to_dict(orient="records")}  # Convert DataFrame to dict format for JSON response

# Run the application using uvicorn (from terminal)
# uvicorn app_name:app --reload
