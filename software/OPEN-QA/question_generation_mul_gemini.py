

import google.generativeai as genai
import os
import json
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import time


genai.configure(api_key="")
model = genai.GenerativeModel("gemini-1.5-flash")
generation_config = {
    'max_output_tokens': 150  # Specify the maximum number of tokens to generate.
}
def generate_questions_from_patent(title, abstract, claim_text):
    """
    Uses the Gemini model to generate questions from a given patent claim.
    """
    prompt = f"""
    You are an expert in patents. Please generate 3 thoughtful, well-crafted questions from the following patent text:
    Title: "{title}"
    Abstract: "{abstract}"
    Claim: "{claim_text}"
    Questions:
    1.
    2.
    3.
    Please don't add any additional text except questions. Please provide no newline between questions only use &&
    """
    response = model.generate_content(
        prompt, generation_config=generation_config
    )
    response_text = response.text
    print(response_text)
    questions = response_text.split("&&")
    questions =  [q.lstrip("1234567890\n") for q in questions]
    questions = [q.strip() for  q in questions]
    print("Total questions",len(questions))
    print(questions)
    return questions

def generate_answers(title, abstract, claim_text, questions):
    """
    Generates answers for the given questions based on the patent data using the Gemini model.
    """
    answers = []
    for question in questions:
        prompt = f"""
        You are a patent expert. Please answer intelligently the following question based on the patent data without any query.
        Title: "{title}"
        Abstract: "{abstract}"
        Claim: "{claim_text}"
        Question: "{question}"
        Answer:
        """
        response = model.generate_content(
            prompt, generation_config=generation_config
        )
        answer = response.text
        answers.append(answer)
    return answers

def generate_qa_pairs(title, abstract, claim):
    questions = generate_questions_from_patent(title, abstract, claim)
    answers = generate_answers(title, abstract, claim, questions)
    return [{"question": question, "answer": answer} for question, answer in zip(questions, answers)]

def process_patent_row(row, save_folder):
    title = row['Patent Name']
    abstract = row['Abstract']
    claim = row['Claim']
    qa_pairs = generate_qa_pairs(title, abstract, claim)

    full_path = os.path.join("data", save_folder)
    os.makedirs(full_path, exist_ok=True)
    file_name = f"{full_path}/{row['Patent ID']}.json"
    with open(file_name, 'w') as file:
        json.dump(qa_pairs, file, indent=4)
    return f"Processed and saved QA pairs for Patent ID: {row['Patent ID']}"

def process_patents(file_path, save_folder="temp", no_of_samples=None):
    start_time = time.time()
    df = pd.read_csv(file_path)

    if no_of_samples is not None:
        df = df.head(no_of_samples)
    # Option 1: Create a new column and filter
    df['word_count'] = df['Abstract'].apply(lambda x: len(str(x).split()))
    filtered_df = df[df['word_count'] >= 100]
    print("Previous length", len(df)) 
    print("New length", len(filtered_df)) 
    df = filtered_df

    data_tuples = [(row, save_folder) for _, row in df.iterrows()]
    with Pool() as pool:
        results = list(tqdm(pool.starmap(process_patent_row, data_tuples), total=len(data_tuples)))
    for result in results:
        print(result)

    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    csv_file_path = 'PatentQADataset.csv'
    process_patents(csv_file_path, "gemini-test", -1)
