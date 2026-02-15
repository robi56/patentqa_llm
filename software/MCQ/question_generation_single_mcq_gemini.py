import pandas as pd
import json
import os
from tqdm import tqdm
import time
import google.generativeai as genai

genai.configure(api_key="")
model = genai.GenerativeModel("gemini-1.5-flash")
generation_config = {'max_output_tokens': 200}

    
def generate_qa_from_patent(patent_id, title, abstract, claim_text):
    """
    Generates multiple-choice questions (MCQs) from a patent claim.
    """
    prompt = f"""
    You are an expert in this area. Read the patent texts (title, abstract, claims) and generate up to 3 challenging multiple-choice questions based on the content. Provide:
    - Question
    - Correct answer
    - Three distractors

    Output as a valid JSON list:
    [
        {{"Patent_ID": "{patent_id}", "Question": "question_text", "Answer": "correct_answer", 
          "Distractors": ["distractor_1", "distractor_2", "distractor_3"]}}
    ]
    Verify the Json string can be load by json.loads function
    """
    
    response = model.generate_content(prompt, generation_config=generation_config)
    response_text = response.text.strip()
    response_text =  response_text.replace("json","").replace("```","")
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response_text}")
        return None

def process_patent_row(row, save_folder):
    result = generate_qa_from_patent(row['Patent ID'], row['Patent Name'], row['Abstract'], row['Claim'])
    os.makedirs(f"data/{save_folder}", exist_ok=True)
    with open(f"data/{save_folder}/{row['Patent ID']}.json", 'w') as file:
        json.dump(result, file, indent=4)
    print(f"Saved QA pairs for Patent ID: {row['Patent ID']}")

def process_patents(file_path, save_folder="temp", no_of_samples=-1):
    start_time = time.time()
    df = pd.read_csv(file_path).head(no_of_samples) if no_of_samples > 0 else pd.read_csv(file_path)
    df = df[df['Abstract'].str.split().str.len() >= 100]
    for _, row in tqdm(df.iterrows(), total=len(df)):
        process_patent_row(row, save_folder)
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    process_patents('PatentQADataset.csv', "gemini-test-mcq", 1)

