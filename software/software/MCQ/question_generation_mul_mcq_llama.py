import pandas as pd
import json
import os
from tqdm import tqdm
from multiprocessing import Pool
import time

import openai
import json
import google.generativeai as genai


from llama_index.llms.together import TogetherLLM

TOGETHERAI_API_KEY = ""
together_llm = TogetherLLM(
    model="meta-llama/Llama-3-8b-chat-hf", api_key=TOGETHERAI_API_KEY
)

# def generate_qa_from_patent(patent_id, title, abstract, claim_text, gpt_model):
#     """
#     Uses GPT-4 to generate multiple-choice questions (MCQs) from a given patent claim.
#     """
#     prompt = f"""
#     You are an expert in this area. Your task is to read the following patent texts and generate up to 3 well-crafted, challenging multiple-choice questions. 
#     A patent text consists of title, abstract, and claims. Each question should be meaningful and grounded in a text snippet.

#     Please provide:
#     - A multiple-choice question
#     - The correct answer
#     - Three distractors (incorrect choices)
    
#     Ensure the questions are non-trivial and properly based on the patent content.

#     ---
#     Patent ID: {patent_id}
#     Title: "{title}"
#     Abstract: "{abstract}"
#     Claim: "{claim_text}"
#     ---

#     The output format should be a valid JSON list:
#     [
#         {{"Patent_ID": "{patent_id}", "Question": "question_text", "Answer": "correct_answer", 
#         "Distractors": ["distractor_1", "distractor_2", "distractor_3"]}}
#     ]
  
#     Provide valid json as output. No extra information.
#     """
#     response = model.generate_content(
#         prompt, generation_config=generation_config
#     )
#     response_text = response.text
#     print(response_text)
#     try:
#         qa_json = json.loads(response_text)
#         return qa_json  # Returning structured JSON data
#     except json.JSONDecodeError:
#         print("Error: Could not decode JSON response from GPT.")
#         return None

def generate_qa_from_patent(patent_id, title, abstract, claim_text, gpt_model):
    """
    Generates multiple-choice questions (MCQs) from a patent claim.
    """
    prompt = f"""
    You are an expert in this area. Your task is to read the following patent texts and generate up to 3 well-crafted, challenging multiple-choice questions. 
    A patent text consists of title, abstract, and claims. Each question should be meaningful and grounded in a text snippet.

    Please provide:
    - A multiple-choice question
    - The correct answer
    - Three distractors (incorrect choices)

    Ensure the questions are non-trivial and properly based on the patent content.

    ---
    Patent ID: {patent_id}
    Title: "{title}"
    Abstract: "{abstract}"
    Claim: "{claim_text}"
    ---

    Output as a valid JSON list:
    [
        {{"Patent_ID": "{patent_id}", "Question": "question_text", "Answer": "correct_answer", 
          "Distractors": ["distractor_1", "distractor_2", "distractor_3"]}}
    ]
    Verify the Json string can be load by json.loads function.  Don't add any additional info expect Json Array.
    """
    response = together_llm.complete(prompt=prompt, max_tokens=300)
    response_text = response.text.strip()
    response_text =  response_text.replace("json","").replace("```","")
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response_text}")
        return None

def generate_qa_pairs(patent_id, title,abstract,  claim, gpt_model):
    """
    Generates a set of QA pairs from a given patent claim.
    """
    questions = generate_questions_from_patent(title,abstract,  claim, gpt_model)
    print(questions)
    answers = generate_answers(title,abstract,  claim, questions, gpt_model)
    # return [{"question": q, "answer": a} for q, a in zip(questions, answers)]

def process_patent_row(row, gpt_model, save_folder, no_of_samples):
    """
    Processes an individual row of patent data.
    """
    patent_id = row['Patent ID']
    title = row['Patent Name']
    abstract = row['Abstract']
    claim = row['Claim']
    result = generate_qa_from_patent(patent_id, title,abstract,  claim, gpt_model)

    # Save to JSON file
    full_path = os.path.join("data", save_folder)
    os.makedirs(full_path, exist_ok=True)
    file_name = f"{full_path}/{row['Patent ID']}.json"

    with open(file_name, 'w') as file:
        json.dump(result, file, indent=4)

    return f"Processed and saved QA pairs for Patent ID: {row['Patent ID']}"

def process_patents(file_path, gpt_model="gpt-4o-mini", save_folder="temp", no_of_samples=-1):
    start_time = time.time()  # Start timing

    df = pd.read_csv(file_path)
    df = df.head(no_of_samples)  # Limit the number of samples if no_of_samples is not -1
    # Option 1: Create a new column and filter
    df['word_count'] = df['Abstract'].apply(lambda x: len(str(x).split()))
    filtered_df = df[df['word_count'] >= 100]
    print("Previous length", len(df)) 
    print("New length", len(filtered_df)) 
    df = filtered_df

    # Prepare data tuples for multiprocessing
    data_tuples = [(row, gpt_model, save_folder, no_of_samples) for index, row in df.iterrows()]

    # Setup multiprocessing pool
    with Pool() as pool:
        results = list(tqdm(pool.starmap(process_patent_row, data_tuples), total=len(data_tuples)))

    for result in results:
        print(result)

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    csv_file_path = 'PatentQADataset.csv'
    models=["gpt-4o", "gpt-4"]
    process_patents(csv_file_path, models[0], "llama-test-mcq", -1)
s