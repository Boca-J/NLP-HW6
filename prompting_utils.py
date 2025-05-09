import os
import json
import re

def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    # TODO
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    return schema

def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # TODO
    # response = response.strip()

    # sql_start = response.lower().find("select")
    # if sql_start == -1:
    #     return ""  

    
    response = response.strip()
    # If inside markdown-style ```sql ... ```
    if "```sql" in response.lower():
        matches = re.findall(r"```sql\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

    # Look for line that starts with SELECT or WITH
    for line in response.splitlines():
        if line.strip().lower().startswith(("select", "with")):
            return line.strip()

    # Look for inline SQL after "SQL:"
    if "SQL:" in response:
        idx = response.upper().find("SQL:")
        return response[idx + 4:].strip()

    # As fallback, find the first "SELECT" or "WITH" in the full text
    for keyword in ("select", "with"):
        idx = response.lower().find(keyword)
        if idx != -1:
            return response[idx:].strip()

    return response[sql_start:].strip()

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")