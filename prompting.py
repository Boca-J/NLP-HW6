import os, argparse, random
from tqdm import tqdm

os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"


import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3ForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig

from utils import set_random_seeds, compute_metrics, save_queries_and_records, compute_records
from prompting_utils import read_schema, extract_sql_query, save_logs
from load_data import load_prompting_data

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # you can add mps

MAX_NEW_TOKENS = 512
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


def create_prompt(sentence, k, examples=None):
    '''
    Function for creating a prompt for zero or few-shot prompting.

    Add/modify the arguments as needed.

    Inputs:
        * sentence (str): A text string
        * k (int): Number of examples in k-shot prompting
    '''
    # TODO
    header = "Translate the following questions into SQL:\n\n"
    prompt = header

    # Few-shot examples
    if k > 0:
        for i in range(min(k, len(examples))):
            nl, sql = examples[i]
            prompt += f"Q: {nl}\nA: {sql}\n\n"

    # The current input
    prompt += f"Q: {sentence}\nA:"

    return prompt


def exp_kshot(tokenizer, model, inputs, k, train_x, train_y):
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
        examples = random.sample(list(zip(train_x, train_y)), k) if k > 0 else None
   
        prompt = create_prompt(sentence, k,examples) # Looking at the prompt may also help
        # print(f"\n==== Prompt for input {i} ====\n{prompt}\n")

        messages=[{
            "role": "system",
            "content": "You are a helpful assistant that generates SQL queries based on natural language instructions.", #you may want to prompt engineer this.
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
            # outputs = model.generate(**input_tokenized, max_new_tokens=MAX_NEW_TOKENS) # You should set MAX_NEW_TOKENS
            outputs = model.generate(...,
                do_sample=False,
                num_beams=1,
                temperature=0.1,
                repetition_penalty=1.2,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True) # How does the response look like? You may need to parse it
        raw_outputs.append(response)

        # Extract the SQL query
        extracted_query = extract_sql_query(response)
        extracted_queries.append(extracted_query)

 

    return raw_outputs, extracted_queries


def eval_outputs(eval_x, eval_y, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluate the outputs of the model by computing the metrics.

    Add/modify the arguments and code as needed.
    '''
    # TODO
    with open(model_sql_path, 'w') as f:
        for query in eval_y:
            f.write(query.strip() + '\n')

    model_queries = eval_y 
    save_queries_and_records(model_queries, model_sql_path, model_record_path)


    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = len(model_error_msgs) / len(model_queries)

    return sql_em, record_em, record_f1, model_error_msgs, error_rate


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
   

    # Model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(model_name, to_quantize)

    for eval_split in ["dev", "test"]:
        eval_x, eval_y = (dev_x, dev_y) if eval_split == "dev" else (test_x, None)

        raw_outputs, extracted_queries = exp_kshot(tokenizer, model, eval_x, shot, train_x, train_y)

        # You can add any post-processing if needed
        # You can compute the records with `compute_records``

        gt_query_records = f"records/{eval_split}_gt_records.pkl"
        gt_sql_path = os.path.join(f'data/{eval_split}.sql')
        gt_record_path = os.path.join(f'records/{eval_split}_gt_records.pkl') 

        #if you saved the records, you can load them here
        model_sql_path = os.path.join(f'results/gemma_{experiment_name}_{eval_split}.sql')
        model_record_path = os.path.join(f'records/gemma_{experiment_name}_{eval_split}.pkl')



        # sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
        #     eval_x, eval_y,
        #     gt_path=gt_sql_path,
        #     model_path=model_sql_path,
        #     gt_query_records=gt_query_records,
        #     model_query_records=model_record_path
        # )

        sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
            eval_x, extracted_queries,
            gt_sql_pth=gt_sql_path,
            model_sql_path=model_sql_path,
            gt_record_path=gt_record_path,
            model_record_path=model_record_path
        )
        print(f"{eval_split} set results: ")
        print(f"Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"{eval_split} set results: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        # Save results
        # You can for instance use the `save_queries_and_records` function

        # Save logs, if needed
        log_path = "" # to specify
        save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)


if __name__ == "__main__":
    main()
