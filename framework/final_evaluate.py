import json
import argparse
import os
import sys
import random

ERROR_FIELDS = [
    "id", "context", "question", "options",
    "answer", "predicted_choice", "fallback_used", "llm_explanation"
]


def evaluate_performance(data, dataset_name):
    true_count = 0
    false_count = 0

    total_records = 0

    # New: per-label correct counts for True / False
    true_answer_correct = 0    # answer == "a" and prediction is correct
    false_answer_correct = 0   # answer == "b" and prediction is correct

    # Record whether each record was predicted correctly, for consecutive-error detection
    error_flags = []        # True means the record was mispredicted
    error_records = []      # one-to-one with data; keeps only the fields needed for output

    for i, record in enumerate(data):
        total_records += 1
        predicted_answer = record.get('predicted_choice', '').strip().lower()
        actual_answer = record.get('answer', '').strip().lower()

        if dataset_name in ["ProntoQA", "Multi-LogiEval"] and predicted_answer.lower() == "c":
            predicted_answer = "b"

        if predicted_answer.lower() == actual_answer.lower():
            true_count += 1
            error_flags.append(False)
            # New: count correct per answer (Unknown excluded)
            if actual_answer == "a":
                true_answer_correct += 1
            elif actual_answer == "b":
                false_answer_correct += 1
        else:
            false_count += 1
            error_flags.append(True)
            print(record.get('id', ''))

        # Extract the fields to save
        error_records.append({field: record.get(field, None) for field in ERROR_FIELDS})

    accuracy = (true_count / total_records) * 100 if total_records > 0 else 0
    return true_count, false_count, total_records, accuracy, error_flags, error_records, true_answer_correct, false_answer_correct


def extract_consecutive_errors(error_flags, error_records):
    """
    Find all consecutive-error segments (runs of True with length >= 2)
    and return all records in these segments.
    """
    consecutive_errors = []
    run = []  # indices of the current consecutive-error run

    for idx, is_error in enumerate(error_flags):
        if is_error:
            run.append(idx)
        else:
            if len(run) >= 1:  # keep only runs of at least 2 consecutive errors
                consecutive_errors.extend(error_records[i] for i in run)
            run = []

    # Handle a consecutive-error run at the end
    if len(run) >= 2:
        consecutive_errors.extend(error_records[i] for i in run)

    return consecutive_errors


def save_consecutive_errors(consecutive_errors, dataset_name):
    save_dir = "./check_error"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(consecutive_errors, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(consecutive_errors)} consecutive error records to: {save_path}")


def compute_f1_score(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def evaluate(result_path, dataset_name, save_errors=False):
    print("Result file: ", result_path)
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    true_count, false_count, total_records, accuracy, error_flags, error_records, true_answer_correct, false_answer_correct = evaluate_performance(data, dataset_name)
    print(f"Total records: {total_records}")
    print(f"Correctly predicted 'true': {true_count}")
    print(f"Correctly predicted 'false': {false_count}")
    print(f"Accuracy: {accuracy:.2f}%")

    # New: print correct ratios for True / False (denominator = dataset size, Unknown excluded)
    true_ratio = (true_answer_correct / total_records) * 100 if total_records > 0 else 0
    false_ratio = (false_answer_correct / total_records) * 100 if total_records > 0 else 0
    print(f"True (answer='A') correct: {true_answer_correct}  ratio: {true_ratio:.2f}%")
    print(f"False (answer='B') correct: {false_answer_correct}  ratio: {false_ratio:.2f}%")

    # Detect and save consecutive errors
    if save_errors:
        consecutive_errors = extract_consecutive_errors(error_flags, error_records)
        if consecutive_errors:
            save_consecutive_errors(consecutive_errors, dataset_name)
        else:
            print("\nNo consecutive errors found.")


if __name__ == "__main__":

    # ========== Switches ==========
    SAVE_CONSECUTIVE_ERRORS = True   # True: save consecutive errors to ./check_error/; False: do not save
    # ==========================

    DATASET_NAME = "FOLIO"  # ProntoQA / ProofWriter / FOLIO / Multi-LogiEval / FLD
    # result_file = os.path.join(f"./Final_results/{DATASET_NAME}_Hybrid_Llama_8B_Qwen.json")  # Hybrid  _Qwen
    # result_file = os.path.join(f"./Final_results_repair/{DATASET_NAME}_Repair_Llama_70B.json")
    result_file = os.path.join(f"./Ablation_results/{DATASET_NAME}_Z3Only.json") # Llama_8B
    evaluate(result_file, DATASET_NAME, save_errors=SAVE_CONSECUTIVE_ERRORS)