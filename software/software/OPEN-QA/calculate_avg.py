import os
import json
import argparse
from collections import defaultdict

def read_and_extract_evaluations(directory):
    """Reads JSON files in the specified directory and extracts evaluation scores."""
    scores = defaultdict(list)
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    for item in data:
                        if 'question' in item:
                            if len(item['question'])<10:
                                continue
                        if 'evaluation' in item and isinstance(item['evaluation'], str):
                            json_string=item['evaluation'].replace('\n', '').replace('    ', '').replace("```",'').replace("json",'')
                            evaluation = json.loads(json_string)
                            for key, value in evaluation.items():
                                if key in ['Relevance', 'Clarity', 'Originality', 'Completeness', 
                                           'Specificity', 'Correctness', 'Consistency'] and isinstance(value, (int, float)):
                                    scores[key].append(value)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file {filepath}")
            except FileNotFoundError:
                print(f"File not found: {filepath}")
            except Exception as e:
                print(f"An error occurred while processing file {filename}: {str(e)}")
    return scores

def calculate_averages(scores):
    """Calculates average scores from the collected scores dictionary."""
    averages = {}
    for key, values in scores.items():
        if values:  # Ensure there are values to avoid division by zero
            averages[key] = sum(values) / len(values)
            print("total questions: ", len(values))
        else:
            averages[key] = 0
    return averages

def main(directory):
    """Main function to handle the workflow."""
    scores = read_and_extract_evaluations(directory)
    averages = calculate_averages(scores)
    print("Average Scores Across All Files:")
    for criterion, average in averages.items():
        print(f"{criterion}: {average:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files to calculate average evaluation scores.')
    parser.add_argument('directory', type=str, help='Directory containing JSON files to process.')
    args = parser.parse_args()
    main(args.directory)

    
#calculate_avg.py data/gpt4-mul-eval 