import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb
import pickle
from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig,T5TokenizerFast
from load_data import load_t5_data, load_lines
from utils import compute_metrics, save_queries_and_records,compute_records



DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--test_batch_size', type=int, default=16)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):

    experiment_name = "greedy"
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):

        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")
        if epoch % 3 == 0:

            eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                            gt_sql_path, model_sql_path,
                                                                            gt_record_path, model_record_path)
            print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
            print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
            if record_f1 > best_f1:
                best_f1 = record_f1
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
   
            save_model(checkpoint_dir, model, best=False)
            if epochs_since_improvement == 0:
                save_model(checkpoint_dir, model, best=True)

            if epochs_since_improvement >= args.patience_epochs:
                break

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):

        

        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    # TODO'

    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
 

    all_generated_sql = []

    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, _, _ in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input[:, :-1]
            )['logits']

            targets = decoder_input[:, 1:]
            non_pad = targets != PAD_IDX
            loss = criterion(logits[non_pad], targets[non_pad])

            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Generate predictions using greedy decoding
            # generation_config = GenerationConfig(
            #     max_new_tokens=256,
            #     do_sample=False  # greedy
            # )
            # outputs = model.generate(
            #     input_ids=encoder_input,
            #     attention_mask=encoder_mask,
            #     generation_config=generation_config,
            #     # eos_token_id=tokenizer.eos_token_id
            # )

            bos = tokenizer.convert_tokens_to_ids('<extra_id_0>')
            outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=1,
                early_stopping=True,
                decoder_start_token_id=bos
            )

            generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            #print(generated)
            all_generated_sql.extend(generated)

    # Save and evaluate
    print("start saving ")
    save_queries_and_records(all_generated_sql, model_sql_path, model_record_path)
    print('Finish Saving')
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(gt_sql_pth, model_sql_path, gt_record_path, model_record_path)
    print('Finish Computing ')
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, record_f1, record_em, sql_em, len(model_error_msgs) / len(dev_loader.dataset)
    
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    print('start inference')
    model.eval()
    all_generated_sql = []

    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            bos = tokenizer.convert_tokens_to_ids('<extra_id_0>')
            outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=1,
                early_stopping=True,
                decoder_start_token_id=bos
            )

            generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print(generated)
            all_generated_sql.extend(generated)

    print('End inference ')
    save_queries_and_records(all_generated_sql, model_sql_path, model_record_path)

def main():

    # gt_sql_path = 'data/dev.sql'
    # gt_record_path = 'records/dev_gt_records.pkl'
    # if not os.path.exists(gt_record_path):
    #     print("Creating dev_gt_records.pkl...")
    #     gt_queries = load_lines(gt_sql_path)
    #     save_queries_and_records(gt_queries, gt_sql_path, gt_record_path)
    

    # with open('records/ground_truth_dev.pkl', 'rb') as f:
    #     records, error_msgs = pickle.load(f)
        

    #     print("✅ Successfully loaded pickle file.")
    #     print(f"# of total queries: {len(records)}")

    #     # Count errors as those with empty records
    #     num_errors = sum(1 for r in records if not r)
    #     num_successes = len(records) - num_errors

    #     print(f"# of successful records: {num_successes}")
    #     print(f"# of SQL errors: {num_errors}")

    #     # Optional: Preview a sample
    #     print("\nSample record:")
    #     print(records[0])
    #     print("\nSample error message (if any):")
    #     print(error_msgs[0] if error_msgs else "No SQL errors.")

    # exit()

    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 

    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_em, dev_record_f1, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
