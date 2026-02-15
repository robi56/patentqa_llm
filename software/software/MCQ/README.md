# MCQ Question Generation

**Run MCQ Generation:**  
```bash
python question_generation_mul_mcq_gemini.py
```
- **Output Directory:** `gemini-test-mcq`
- **Sample Output Format:**
```json
[
  {
    "Patent_ID": "12219595",
    "Question": "According to Claim 1, under what condition is the transmission of second signaling and/or second information omitted?",
    "Answer": "When the second resource structure has a lower priority than the first and third resource structures and overlaps with at least one of them.",
    "Distractors": [
      "When the first and third resource structures are fully utilized.",
      "When the second resource structure is completely disjoint.",
      "When the wireless device experiences interference."
    ]
  }
]
```

---

# Evaluation and Scoring

**Run Evaluation:**  
```bash
python eval_qa_paris_gemini_mcq.py \
  --input_json_directory data/gemini-test-mcq \
  --input_csv_path PatentQADataset.csv \
  --output_json_directory data/gemini-test-mcq-eval-gemini/
```
- **Output Directory:** `data/gemini-test-mcq-eval-gemini/`
- **Sample Evaluation Format:**
```json
{
  "Patent_ID": "12219595",
  "Question": "According to Claim 1, under what condition is the transmission omitted?",
  "evaluation": {
    "Groundedness": 10,
    "Groudedness_Justification": "Question directly references Claim 1's conditions.",
    "Correctness": 10,
    "Correctness_Justification": "Answer matches the claim’s stated conditions.",
    "Quality of Distractors": 8,
    "Distractor_Quality_Justification": "Plausible alternatives, though one is overly generic."
  }
}
```

---

# Average Score Calculation

**Run Scoring Script:**  
```bash
python calculate_avg_mcq.py gemini-test-mcq-eval-gemini/
```
- **Purpose:** Computes average scores (e.g., Groundedness, Correctness, Distractor Quality) across all generated questions.

---
This README provides a concise workflow from question generation to evaluation and scoring. Let me know if you need any adjustments or additional sections.
