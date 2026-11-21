
import json
import csv
import requests
from datetime import datetime

MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"

with open("golden_set.json", "r", encoding="utf-8") as file:
    questions = json.load(file)

results = []

print("\n===================================")
print(" Llama 3.2 - Baseline Evaluation")
print(" Release v0.1")
print("===================================\n")

for item in questions:
    question_id = item["id"]
    question = item["question"]
    expected_answer = item["expected_answer"]
    keywords = item["keywords"]

    print(f"Evaluating {question_id}: {question}")

    prompt = f"""
    Answer the following question clearly and accurately.

    Question: {question}

    Give a concise answer.
    """

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=300
        )

        response.raise_for_status()
        answer = response.json()["response"].strip()

        matched_keywords = [
            keyword for keyword in keywords
            if keyword.lower() in answer.lower()
        ]

        score = (len(matched_keywords) / len(keywords)) * 100

        status = "PASS" if score >= 60 else "REVIEW"

        results.append({
            "id": question_id,
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": answer,
            "matched_keywords": ", ".join(matched_keywords),
            "score": round(score, 2),
            "status": status,
            "source": item["source"]
        })

        print(f"Score: {score:.2f}% | Status: {status}\n")

    except requests.exceptions.RequestException as error:
        print(f"Error evaluating {question_id}: {error}\n")

        results.append({
            "id": question_id,
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": "",
            "matched_keywords": "",
            "score": 0,
            "status": "ERROR",
            "source": item["source"]
        })

with open("baseline_results.csv", "w", newline="", encoding="utf-8") as file:
    fieldnames = [
        "id",
        "question",
        "expected_answer",
        "generated_answer",
        "matched_keywords",
        "score",
        "status",
        "source"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

total = len(results)
passed = sum(1 for result in results if result["status"] == "PASS")
review = sum(1 for result in results if result["status"] == "REVIEW")
errors = sum(1 for result in results if result["status"] == "ERROR")

print("\n===================================")
print(" EVALUATION SUMMARY")
print("===================================")
print(f"Model: {MODEL}")
print(f"Release: v0.1")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total Questions: {total}")
print(f"PASS: {passed}")
print(f"REVIEW: {review}")
print(f"ERROR: {errors}")
print("Results saved to baseline_results.csv")
print("===================================")
