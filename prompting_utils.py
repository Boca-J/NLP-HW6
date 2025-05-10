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
    markdown_match = re.search(r"```sql\s*(.*?)```", response, re.IGNORECASE | re.DOTALL)
    if markdown_match:
        return markdown_match.group(1).strip()

    # 2. Look for "SQL:" prefix
    if "SQL:" in response.upper():
        sql_part = response.split("SQL:")[-1].strip()
        if sql_part.lower().startswith(("select", "with")):
            return sql_part

    # 3. Find any line that starts with SELECT or WITH
    for line in response.splitlines():
        if line.strip().lower().startswith(("select", "with")):
            return line.strip()

    # 4. Fallback: locate first "select"/"with" keyword in entire text
    match = re.search(r"(select|with)\s.+", response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()

    # 5. If nothing found, return empty string
    return ""

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")