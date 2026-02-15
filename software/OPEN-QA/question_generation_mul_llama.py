import pandas as pd
import json
import os
from tqdm import tqdm
from multiprocessing import Pool
import time
import openai

from llama_index.llms.together import TogetherLLM

TOGETHERAI_API_KEY = ""
together_llm = TogetherLLM(
    model="meta-llama/Llama-3-8b-chat-hf", api_key=TOGETHERAI_API_KEY
)


# Assuming client is properly configured for OpenAI API access
# import client setup here
# Configure the OpenAI client with your API key
openai.api_key = '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'
client = openai.OpenAI(api_key="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")


def generate_questions_from_patent(title, abstract, claim_text):
    """
    Uses the model to generate questions from a given patent claim.
    """
    prompt = f"""
    You are an expert in patent. Please generate 3 thoughtful, well-crafted questions from the following patent text:
    Title: "{title}"
    Abstract: "{abstract}"
    Claim: "{claim_text}"
    
    Questions:
    1.
    2.
    3.
    Please don't add any additional text expect questions. Don't provide any additional newlines between questions. 
    """
    response = together_llm.complete(prompt=prompt, max_tokens=150)
    response_text = response.text
    print(response_text)
    questions = response_text.split("\n")
    questions =  [q.lstrip("1234567890. ") for q in questions]
    print(questions)
    return questions

def generate_answers(title, abstract, claim_text, questions):
    """
    Generates answers for the given questions based on the patent data using the model.
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
        response = together_llm.complete(prompt=prompt, max_tokens=150)
        answer = response.text 
        answers.append(answer)
    return answers

def generate_qa_pairs(title, abstract, claim):
    """
    Generates a set of QA pairs from a given patent claim.
    """
    questions = generate_questions_from_patent(title, abstract, claim)
    print("total questions: ", len(questions))
    answers = generate_answers(title, abstract, claim, questions)
    return [{"question": question, "answer": answer} for question, answer in zip(questions, answers)]

def process_patent_row(row, save_folder):
    """
    Processes an individual row of patent data.
    """
    title = row['Patent Name']
    abstract = row['Abstract']
    claim = row['Claim']
    qa_pairs = generate_qa_pairs(title, abstract, claim)

    # Save to JSON file
    full_path = os.path.join("data", save_folder)
    os.makedirs(full_path, exist_ok=True)
    file_name = f"{full_path}/{row['Patent ID']}.json"
    with open(file_name, 'w') as file:
        json.dump(qa_pairs, file, indent=4)

    return f"Processed and saved QA pairs for Patent ID: {row['Patent ID']}"

def process_patents(file_path, save_folder="temp", no_of_samples=None):
    start_time = time.time()  # Start timing

    df = pd.read_csv(file_path)
    if no_of_samples is not None:
        df = df.head(no_of_samples)  # Limit the number of samples

    # Option 1: Create a new column and filter
    df['word_count'] = df['Abstract'].apply(lambda x: len(str(x).split()))
    filtered_df = df[df['word_count'] >= 100]
    print("Previous length", len(df)) 
    print("New length", len(filtered_df)) 
    df = filtered_df

    # Prepare data tuples for multiprocessing
    data_tuples = [(row, save_folder) for _, row in df.iterrows()]

    # Setup multiprocessing pool
    with Pool() as pool:
        results = list(tqdm(pool.starmap(process_patent_row, data_tuples), total=len(data_tuples)))

    for result in results:
        print(result)

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    csv_file_path = 'PatentQADataset.csv'
    process_patents(csv_file_path, "llama3-test", -1)