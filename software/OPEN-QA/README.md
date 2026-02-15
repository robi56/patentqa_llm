# Question Answer Generation

## Step 1: Generate Questions and Answers
Run the following script to generate questions and answers from the patent descriptions:
```bash
python question_generation_mul_gemini.py
```
Example output:
```json
[
  {
    "question": "What specific measures are in place to ensure the synchronization of audio signals...",
    "answer": "The provided patent application does not detail specific measures to ensure synchronization..."
  }
]
```

---

## Step 2: Evaluation and Scoring
Run the evaluation script to score the generated Q&A pairs against a provided dataset:
```bash
python eval_qa_paris_mul.py \
  --input_json_directory data/gemini-test/ \
  --input_csv_path PatentQADataset.csv \
  --output_json_directory data/gemini-test-eval/
```
Example evaluation output:
```json
[
  {
    "question": "What specific measures are in place to ensure the synchronization of audio signals...",
    "answer": "The provided patent application does not detail specific measures...",
    "evaluation": {
      "Relevance": 9,
      "Clarity": 10,
      "Originality": 10,
      "Completeness": 9,
      "Specificity": 8,
      "Correctness": 10,
      "Consistency": 10,
      "Justification": "The question is very relevant..."
    }
  }
]
```

---

## Step 3: Calculate Average Scores
To compute the average evaluation scores across all Q&A pairs, run the following script:
```bash
python calculate_avg.py data/gemini-test-eval/
```
The output will display average scores for all criteria (e.g., Relevance, Clarity, Originality, etc.) in the console.

---

## Directory Structure
```
project-root/
├── data/
│   ├── gemini-test/
│   │   └── qa_pairs.json
│   └── gemini-test-eval/
│       └── eval_results.json
├── PatentQADataset.csv
├── question_generation_mul_gemini.py
├── eval_qa_paris_mul.py
└── calculate_avg.py
```
---

## Requirements
- Python 3.10+
- Install dependencies with:
```bash
pip install -r requirements.txt
```

## Notes
- Ensure `PatentQADataset.csv` is in the root directory.
- Adjust script parameters as needed for different datasets.
