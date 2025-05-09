import os, argparse, random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3ForConditionalGeneration
from transformers import BitsAndBytesConfig

from utils import set_random_seeds, compute_metrics, save_queries_and_records, compute_records
from prompting_utils import read_schema, extract_sql_query, save_logs
from load_data import load_prompting_data



DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # you can add mps
MAX_NEW_TOKENS = 256

def get_args():
    '''
    Arguments for prompting. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(
        description='Text-to-SQL experiments with prompting.')

    parser.add_argument('-s', '--shot', type=int, default=0,
                        help='Number of examples for k-shot learning (0 for zero-shot)')
    parser.add_argument('-m', '--model', type=str, default='gemma-1b',
                        help='Model to use for prompting')
    parser.add_argument('-q', '--quantization', action='store_true',
                        help='Use a quantized version of the model (e.g. 4bits)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to help reproducibility')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")
    args = parser.parse_args()
    return args


def create_prompt(sentence, k, examples=None, schema_info=None):
    '''
    Function for creating a prompt for zero or few-shot prompting.

    Inputs:
        * sentence (str): A natural language query to convert to SQL
        * k (int): Number of examples in k-shot prompting
        * examples (list): Optional list of (nl_query, sql_query) pairs for few-shot learning
        * schema_info (str): Optional database schema information
    
    Returns:
        * prompt (str): The complete prompt for the model
    '''
    # Default schema information if not provided
    if schema_info is None:
        schema_info = """
    Database Schema:
    - flight: Contains flight_id, from_airport, to_airport, airline_code, departure_time, arrival_time
    - airport: Contains airport_code, city_code, airport_name
    - city: Contains city_code, city_name, country_name
    - fare: Contains fare_id, flight_id, fare_code, amount, restrictions
    - ground_transport: Contains city_code, transport_type, price
    - airport_service: Contains airport_code, city_code, miles_distant, direction, minutes_distant
    """

    # Create base instruction
    prompt = f"""You are a SQL expert that converts natural language questions into SQL queries for a flight database.
    {schema_info}

    IMPORTANT INSTRUCTIONS:
    1. Generate ONLY the SQL query without any additional text, explanation, or tags
    2. Do not include <end_of_turn> or any other special tokens
    3. The query must be a valid SQL query that can be executed directly
    4. Always start with SELECT and end with the complete query

    I will give you a question about flights, and you need to generate a valid SQL query that answers the question.
    """

    # Add few-shot examples if k > 0 and examples are provided
    if k > 0 and examples is not None:
        prompt += "\n\nHere are some examples:\n"
        
        # Use minimum of k or available examples
        for i in range(min(k, len(examples))):
            nl, sql = examples[i]
            prompt += f"\nQuestion: {nl}\nSQL: {sql}\n"
    
    # Add the current query
    prompt += f"\nQuestion: {sentence}\nSQL: "
    
    return prompt


def exp_kshot(tokenizer, model, inputs, k):
    '''
    k-shot prompting experiments using the provided model and tokenizer. 
    This function generates SQL queries from text prompts and evaluates their accuracy.

    Add/modify the arguments and code as needed.

    Inputs:
        * tokenizer
        * model
        * inputs (List[str]): A list of text strings
        * k (int): Number of examples in k-shot prompting
    '''
    raw_outputs = []
    extracted_queries = []

    for i, sentence in tqdm(enumerate(inputs)):
        prompt = create_prompt(sentence, k) # Looking at the prompt may also help


        messages=[{
          "role": "system",
          "content": "You are an expert SQL query generator. You translate natural language questions into SQL queries for a flight database. Your answers contain ONLY the SQL query without any explanations, comments, or formatting tags.",
              },
        {
            "role": "user",
            "content": prompt,
        }
        ]
        input_tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
          outputs = model.generate(
            **input_tokenized, 
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Use greedy decoding for deterministic results
            num_beams=1,      # Use beam search with 1 beam (equivalent to greedy)
            temperature=0.1,  # Low temperature for more focused responses
            repetition_penalty=1.2,  # Discourage repetitions like <end_of_turn>
            use_cache= False,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Add skip_special_tokens=True

        # Extract the SQL query
        extracted_query = extract_sql_query(response)
        print(f"[{i+1}] NL Question: {sentence}")
        print(f"[{i+1}] Generated SQL: {extracted_query}\n")
        extracted_queries.append(extracted_query)


 

    return raw_outputs, extracted_queries


def eval_outputs(extracted_queries, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluate the outputs of the model by computing the metrics.
    
    Inputs:
        * extracted_queries (List[str]): Generated SQL queries from the model
        * gt_sql_pth (str): Path to ground truth SQL queries
        * model_sql_path (str): Path to save model generated SQL queries
        * gt_record_path (str): Path to ground truth database records
        * model_record_path (str): Path to save model generated records
        
    Returns:
        * sql_em (float): SQL exact match accuracy
        * record_em (float): Record exact match accuracy
        * record_f1 (float): F1 score for records
        * model_error_msgs (List[str]): Error messages from query execution
        * error_rate (float): Rate of SQL syntax errors
    '''
    # Save the model generated queries to a file
    with open(model_sql_path, 'w') as f:
        for query in extracted_queries:
            f.write(f"{query}\n")
    
    # Compute database records for the generated queries
    records, error_msgs = compute_records(extracted_queries)
    
    # Save the computed records
    with open(model_record_path, 'wb') as f:
        import pickle
        pickle.dump((records, error_msgs), f)
    
    # Compute metrics using the utility function
    sql_em, record_em, record_f1, model_error_rate = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    
    # Calculate error rate (percentage of queries with syntax errors)
    error_count = sum(1 for msg in error_msgs if msg != "")
    error_rate = error_count / len(extracted_queries) if extracted_queries else 0
    
    print(f"SQL EM: {sql_em:.4f}")
    print(f"Record EM: {record_em:.4f}")
    print(f"Record F1: {record_f1:.4f}")
    print(f"Error rate: {error_rate:.4f}")
    
    return sql_em, record_em, record_f1, error_msgs, error_rate


def initialize_model_and_tokenizer(model_name, to_quantize=False):
    '''
    Args:
        * model_name (str): Model name (e.g., "gemma-1b").
        * to_quantize (bool): Use a quantized version of the model (e.g. 4bits)
    
    To access to the model on HuggingFace, you need to log in and review the 
    conditions and access the model's content.
    '''
    if model_name == "gemma-1b":
        # model_id = "google/gemma-3-1b-it"
        model_id = "google/gemma-3-1b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Native weights exported in bfloat16 precision, but you can use a different precision if needed
        
        if to_quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # 4-bit quantization
            )
            model = Gemma3ForCausalLM.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16,
                                                        config=nf4_config).to(DEVICE)
        else:
            model = Gemma3ForCausalLM.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16).to(DEVICE)
    elif model_name == "gemma-27b":
        model_id = "google/gemma-3-27b-it"

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()

        processor = AutoProcessor.from_pretrained(model_id)


    
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented in this template.")
        # #you can extend this to use 4B and 12B versions. 


    return tokenizer, model


def main():
    '''
    Note: this code serves as a basic template for the prompting task. You can but 
    are not required to use this pipeline.
    You can design your own pipeline, and you can also modify the code below.
    '''
    args = get_args()
    shot = args.shot
    model_name = args.model
    to_quantize = args.quantization
    experiment_name = args.experiment_name

    set_random_seeds(args.seed)

    data_folder = 'data'
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)
    print(f"Dataset sizes - Train: {len(train_x)}, Dev: {len(dev_x)}, Test: {len(test_x)}")

    # Model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(model_name, to_quantize)

    for eval_split in ["dev", "test"]:
        eval_x, eval_y = (dev_x, dev_y) if eval_split == "dev" else (test_x, None)
        
        # Generate SQL queries
        raw_outputs, extracted_queries = exp_kshot(tokenizer, model, eval_x, shot)
        
        # Create output paths
        model_sql_path = os.path.join(f'results/gemma_{shot}shot_{experiment_name}_{eval_split}.sql')
        model_record_path = os.path.join(f'records/gemma_{shot}shot_{experiment_name}_{eval_split}.pkl')
        
        # For dev set, we can evaluate because we have ground truth
        if eval_split == "dev":
            gt_sql_path = os.path.join('data/dev.sql')
            gt_record_path = os.path.join('records/ground_truth_dev.pkl')
            
            # Evaluate on dev set
            sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
                extracted_queries,
                gt_sql_path,
                model_sql_path,
                gt_record_path,
                model_record_path
            )
            
            print(f"Dev set results:")
            print(f"Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
            print(f"Dev set: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
            
            # Save logs
            log_path = f"logs/gemma_{shot}shot_{experiment_name}_dev.log"
            save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)
            
        # For test set, just save the queries and generate records
        else:  # eval_split == "test"
            # Save the model generated queries to a file
            with open(model_sql_path, 'w') as f:
                for query in extracted_queries:
                    f.write(f"{query}\n")
            
            # Compute database records for the generated queries
            records, error_msgs = compute_records(extracted_queries)
            
            # Save the computed records
            with open(model_record_path, 'wb') as f:
                import pickle
                pickle.dump((records, error_msgs), f)
            
            # Calculate error rate (just for logging)
            error_count = sum(1 for msg in error_msgs if msg != "")
            error_rate = error_count / len(extracted_queries) if extracted_queries else 0
            print(f"Test set: Generated {len(extracted_queries)} SQL queries")
            print(f"Test set: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
            
            # Create copies with the simplified names required for submission
            final_sql_path = os.path.join('results/gemma_test.sql')
            final_record_path = os.path.join('records/gemma_test.pkl')
            
            # Copy the files
            import shutil
            shutil.copy(model_sql_path, final_sql_path)
            shutil.copy(model_record_path, final_record_path)
            
            print(f"Final submission files saved to {final_sql_path} and {final_record_path}")

if __name__ == "__main__":
    main()
