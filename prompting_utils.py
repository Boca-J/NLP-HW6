import os
import re

def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    if not os.path.exists(schema_path):
        # Return a default schema if file doesn't exist
        return """
Database Schema:
- flight: Contains flight_id, from_airport, to_airport, airline_code, departure_time, arrival_time
- airport: Contains airport_code, city_code, airport_name
- city: Contains city_code, city_name, country_name
- fare: Contains fare_id, flight_id, fare_code, amount, restrictions
- ground_transport: Contains city_code, transport_type, price
- airport_service: Contains airport_code, city_code, miles_distant, direction, minutes_distant
"""
    
    try:
        with open(schema_path, 'r') as f:
            schema_content = f.read()
        return schema_content
    except Exception as e:
        print(f"Error reading schema file: {e}")
        # Return a default schema if there's an error
        return """
Database Schema:
- flight: Contains flight_id, from_airport, to_airport, airline_code, departure_time, arrival_time
- airport: Contains airport_code, city_code, airport_name
- city: Contains city_code, city_name, country_name
- fare: Contains fare_id, flight_id, fare_code, amount, restrictions
- ground_transport: Contains city_code, transport_type, price
- airport_service: Contains airport_code, city_code, miles_distant, direction, minutes_distant
"""

def extract_sql_query(response):
    """
    Extract the SQL query from the model response.
    The response may include text like 'SQL: SELECT ...'
    or markdown blocks. This function isolates the SQL.
    """
    # Strip leading/trailing whitespace
    response = response.strip()

    # Remove markdown formatting if present
    if "```sql" in response:
        # Grab content inside code block
        matches = re.findall(r"```sql\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

    # Try to find line starting with SELECT or WITH
    lines = response.split("\n")
    for line in lines:
        if line.strip().lower().startswith(("select", "with")):
            return line.strip()

    # If response is a single SQL query with no prefix
    if response.lower().startswith(("select", "with")):
        return response

    # Try to isolate after "SQL: "
    if "SQL:" in response:
        return response.split("SQL:")[-1].strip()

    # Fallback: return raw response and log a warning
    print(f"[Warning] Could not extract SQL. Raw response: {response}")
    return response
    
def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"SQL Exact Match: {sql_em:.4f}\n")
        f.write(f"Record Exact Match: {record_em:.4f}\n")
        f.write(f"Record F1 Score: {record_f1:.4f}\n")
        
        # Calculate error rate
        error_count = sum(1 for msg in error_msgs if msg != "")
        error_rate = error_count / len(error_msgs) if error_msgs else 0
        f.write(f"Error Rate: {error_rate:.4f}\n\n")
        
        # Log common error types
        error_types = {}
        for msg in error_msgs:
            if msg == "":
                continue
            error_type = msg.split(":")[0] if ":" in msg else msg
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        f.write("Common Error Types:\n")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {error_type}: {count}\n")
            
        # Include some example errors
        f.write("\nSample Error Messages:\n")
        error_samples = [msg for msg in error_msgs if msg != ""][:5]  # First 5 errors
        for i, error in enumerate(error_samples):
            f.write(f"Error {i+1}: {error}\n")