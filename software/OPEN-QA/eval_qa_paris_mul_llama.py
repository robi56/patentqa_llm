

import json
import os
import pandas as pd
from tqdm import tqdm
import argparse
import os 
import openai
import json
from multiprocessing import Pool, Manager
import google.generativeai as genai
from llama_index.llms.together import TogetherLLM

TOGETHERAI_API_KEY = ""
together_llm = TogetherLLM(
    model="meta-llama/Llama-3-8b-chat-hf", api_key=TOGETHERAI_API_KEY
)

# Assuming evaluate_qa_pair and other required modules are already imported

def evaluate_qa_pair(question, answer, title, claim_text, abstract_text):
    """
    Evaluates the quality of a QA pair based on multiple dimensions in the context of patent claims and abstracts.
    """
    # You can switch between claim_text and abstract_text depending on your needs
    patent_text = f"{title} {claim_text}"  # or use f"{title} {abstract_text}" as needed

    prompt = f"""
    You are an expert in this area. Your task is to evaluate the quality of question-answer pairs in the context of the patent text provided.

    Given the following patent text:
    "{patent_text}"

    Evaluate the following question and answer:
    Question: "{question}"
    Answer: "{answer}"

    Provide a score (1-10) for the following quality metrics:
    - *Relevance:* How well the question relates to the patent text.
    - *Clarity:* How clear and understandable the question and answer are?
    - *Originality (Non-Verbatim Reproduction):* Does the answer avoid simply copying sections of the patent text and instead provide a meaningful synthesis?
    - *Completeness:* Does the answer fully address all aspects of the question?
    - *Specificity:* Does the answer precisely address the question, avoiding vagueness or overgeneralization?
    - *Correctness:* Whether the answer is factually correct based on the text provided.
    - *Consistency:* Is the answer logically consistent with the patent text and free from contradictions?

    *Response Format (JSON)*:
    {{
        "Relevance": <score>,
        "Clarity": <score>,
        "Originality": <score>,
        "Completeness": <score>,
        "Specificity": <score>,
        "Correctness": <score>,
        "Consistency": <score>,
    }}
    
    Provide simple json format as output. No extra information. You need not add  ```json at first.
    """

    response = together_llm.complete(prompt=prompt, max_tokens=150)
    response_text = response.text
    evaluation = response_text.strip()
    return evaluation





def read_json_files(directory):
    """ Reads all JSON files in the specified directory. """
    qa_data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                patent_id = filename.split('.')[0]  # Assuming filename is the patent ID
                qa_data[patent_id] = json.load(file)
    return qa_data

def read_csv_file(csv_file_path):
    """ Reads CSV file containing patent details. """
    return pd.read_csv(csv_file_path)


def evaluate_qa_pair_wrapper(args):
    # Assuming evaluate_qa_pair function handles the evaluation logic
    return evaluate_qa_pair(*args)


def evaluate_and_save(qa_data, patent_details, output_directory):
    """
    Evaluates each QA pair, adds scores, and saves the results to new JSON files using multiprocessing.
    """
    os.makedirs(output_directory, exist_ok=True)
    
    # Setup multiprocessing pool
    pool = Pool(processes=os.cpu_count())  # Create a Pool with cpu_count processes

    results = []
    tasks = []
    
    for patent_id, qas in qa_data.items():
        row = patent_details[patent_details['Patent ID'] == patent_id]
        if row.empty:
            continue
        
        title = row.iloc[0]['Patent Name']
        claim_text = row.iloc[0]['Claim']
        abstract_text = row.iloc[0]['Abstract']
        
        for qa in qas:
            tasks.append((qa['question'], qa['answer'], title, claim_text, abstract_text))
    
    # Map tasks to the pool
    evaluated_qas = pool.map(evaluate_qa_pair_wrapper, tasks)
    pool.close()  # Close the pool
    pool.join()   # Wait for all worker processes to finish
    
    # Organize results back to their respective patents and save
    idx = 0
    for patent_id, qas in qa_data.items():
        for qa in qas:
            qa['evaluation'] = evaluated_qas[idx]
            idx += 1
        
        # Save the evaluated QA pairs
        with open(os.path.join(output_directory, f"{patent_id}.json"), 'w') as file:
            json.dump(qas, file, indent=4)
            
    return

def main(args):
    # Read JSON files with QA pairs
    qa_pairs = read_json_files(args.input_json_directory)

    # Read CSV file with patent details
    patent_details = read_csv_file(args.input_csv_path)
    
    print(len(qa_pairs))

    # Evaluate QA pairs and save results
    evaluate_and_save(qa_pairs, patent_details, args.output_json_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and evaluate patent QA pairs.')
    parser.add_argument('--input_json_directory', type=str, required=True, help='Directory containing JSON files with QA pairs.')
    parser.add_argument('--input_csv_path', type=str, required=True, help='CSV file path containing patent details.')
    parser.add_argument('--output_json_directory', type=str, required=True, help='Output directory to save evaluated QA pairs.')
    
    args = parser.parse_args()
    main(args)

    
#python eval_qa_pairs_mul.py --input_json_directory data/gpt4-mul --input_csv_path PatentQADataset.csv --output_json_directory data/gpt4-mul-eval


