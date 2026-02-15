import pandas as pd
import json
import os
from tqdm import tqdm
from multiprocessing import Pool
import time
import openai


client = openai.OpenAI(api_key="")

def generate_questions_from_patent(title, abstract, claim_text,gpt_model):
    """
    Uses GPT-4 to generate questions from a given patent claim.
    """
    prompt = f"""
    You are an expert in patent. Please generate 3 thoughtful, well-crafted questions from the following patent text:
    
    Title: "{ title}"
    Abstract: "{abstract}"
    Claim: "{claim_text}"

    
    
    Questions:
    1.
    2.
    3.
    
    Please don't add any additional text except questions. Please provide no newline between questions only use &&
    """

    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You generate questions from patent claims."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
    )

    questions = response.choices[0].message.content.strip()
    questions = questions.split("&&")
    questions =  [q.lstrip("1234567890\n") for q in questions]
    questions = [q.strip() for  q in questions]
    print("Total questions",len(questions))
    print(questions)
    return questions
    return [q.lstrip("1234567890. ") for q in questions]


def generate_answers(title,abstract,  claim_text, questions,gpt_model):
    """
    Uses GPT-4 to generate answers for the given questions based on the patent claim.
    """
    answers = []
    for question in questions:
        prompt = f"""
         You are a patent expert. Please answer intelligently the following question based on the patent data without any query.
         
        Title: "{ title}"
        Abstract: "{abstract}"
        Claim: "{claim_text}"


        Question: "{question}"

        Answer:
        """

        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You generate answers from patent claims."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
        )

        answer = response.choices[0].message.content.strip()
        answers.append(answer)

    return answers


def generate_qa_pairs(title,abstract,  claim, gpt_model):
    """
    Generates a set of QA pairs from a given patent claim.
    """
    questions = generate_questions_from_patent(title,abstract,  claim, gpt_model)
    answers = generate_answers(title,abstract,  claim, questions, gpt_model)
    return [{"question": q, "answer": a} for q, a in zip(questions, answers)]

def process_patent_row(row, gpt_model, save_folder, no_of_samples):
    """
    Processes an individual row of patent data.
    """
    patent_claim = f"{row['Abstract']} {row['Patent Name']} {row['Claim']}"
    title = row['Patent Name']
    abstract = row['Abstract']
    claim = row['Claim']
    qa_pairs = generate_qa_pairs(title,abstract,  claim, gpt_model)

    # Save to JSON file
    full_path = os.path.join("data", save_folder)
    os.makedirs(full_path, exist_ok=True)
    file_name = f"{full_path}/{row['Patent ID']}.json"

    with open(file_name, 'w') as file:
        json.dump(qa_pairs, file, indent=4)

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
    process_patents(csv_file_path, models[0], "gpt-4o-test", -1)
