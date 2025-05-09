import os
import json

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
    response = response.strip()

    sql_start = response.lower().find("select")
    if sql_start == -1:
        return ""  

   
    return response[sql_start:].strip()

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")