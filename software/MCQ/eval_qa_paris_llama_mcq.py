import json
import os
import pandas as pd
import argparse
import openai
from tqdm import tqdm

from llama_index.llms.together import TogetherLLM


TOGETHERAI_API_KEY = ""
together_llm = TogetherLLM(
    model="meta-llama/Llama-3-8b-chat-hf", api_key=TOGETHERAI_API_KEY
)


def evaluate_qa_pair(question, answer, title, claim_text, abstract_text,distractors):
    """
    Evaluates the quality of a QA pair based on multiple dimensions in the context of patent claims and abstracts.
    """
    # You can switch between claim_text and abstract_text depending on your needs
    patent_text = f"{title} {abstract_text} {claim_text}"  # or use f"{title} {abstract_text}" as needed
    prompt = f"""
    You are an expert in this area. Your task is to evaluate the quality of the generated question, answer, and distractors based on a given patent text. Below is the provided information:

    Question: {question}
    Context (based on which the question has been generated): {patent_text}
    Answer: {answer}
    Distractors: {distractors}

    Please assess the following criteria on a scale of 1 to 10:

    - Groundedness: How well the question is grounded in the provided text snippet.
    - Correctness: Whether the answer is accurate and directly inferable from the text snippet.
    - Quality of Distractors: How good the distractors are.

    *Response Format (JSON)*:
    {{
     "Groundedness": <score>, 
     "Correctness": <score>, 
     "Quality of Distractors": <score>
    }}

    Provide the output in a simple JSON format. No extra information. You need not add ```json at the beginning. Please provide valid JSON format.
    """

    response = together_llm.complete(prompt=prompt, max_tokens=250)
    response_text = response.text
    evaluation = json.loads(response_text)
    return evaluation 



def load_json(directory):
    return {file.split('.')[0]: json.load(open(os.path.join(directory, file)))
            for file in os.listdir(directory) if file.endswith('.json')}

def load_csv(path):
    return pd.read_csv(path)

def evaluate_and_save(qa_data, patents, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for patent_id, qas in tqdm(qa_data.items()):
        row = patents.loc[patents['Patent ID'] == patent_id]
        if row.empty or not qas:
            continue

        title, claim_text, abstract_text = row.iloc[0][['Patent Name', 'Claim', 'Abstract']]
        try: 
            for qa in qas:
                qa['evaluation'] = evaluate_qa_pair(
                    qa['Question'], qa['Answer'], title, claim_text, abstract_text, qa.get("Distractors", "")
                )

            with open(os.path.join(output_dir, f"{patent_id}.json"), 'w') as file:
                json.dump(qas, file, indent=4)
        except Exception as e:
            print(e, qas)
def main(args):
    qa_pairs = load_json(args.input_json_directory)
    patents = load_csv(args.input_csv_path)
    print(f"Evaluating {len(qa_pairs)} QA pairs")
    evaluate_and_save(qa_pairs, patents, args.output_json_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate patent QA pairs without multiprocessing.')
    parser.add_argument('--input_json_directory', type=str, required=True)
    parser.add_argument('--input_csv_path', type=str, required=True)
    parser.add_argument('--output_json_directory', type=str, required=True)
    
    args = parser.parse_args()
    main(args)
