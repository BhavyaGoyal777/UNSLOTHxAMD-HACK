#!/usr/bin/env python3
"""
Interactive script to upload curated dataset to Hugging Face
"""

import json
from pathlib import Path
from datasets import Dataset

def load_all_questions():
    """Load all curated questions from JSON files"""
    CURATED_DIR = Path("MAIN_CURATED_JSON")
    all_questions = []
    curated_files = sorted(CURATED_DIR.glob("*.json"))

    for file_path in curated_files:
        try:
            with open(file_path, 'r') as f:
                questions = json.load(f)

                if isinstance(questions, list):
                    for q in questions:
                        if validate_question(q):
                            all_questions.append(clean_question(q))
                elif isinstance(questions, dict):
                    if validate_question(questions):
                        all_questions.append(clean_question(questions))
        except:
            pass

    return all_questions

def validate_question(q):
    """Validate that question has required fields"""
    if not isinstance(q, dict):
        return False

    required_fields = ['topic', 'question', 'choices', 'answer']

    # Check required fields
    if not all(field in q for field in required_fields):
        return False

    # Validate choices format
    if not isinstance(q.get('choices'), list) or len(q.get('choices')) != 4:
        return False

    # Validate answer format
    if q.get('answer') not in ['A', 'B', 'C', 'D']:
        return False

    return True

def clean_question(q):
    """Clean question and remove difficulty field"""
    cleaned = {
        "topic": q['topic'],
        "question": q['question'],
        "choices": q['choices'],
        "answer": q['answer'],
        "explanation": q.get('explanation', ''),
        "reasoning": q.get('reasoning', '')
    }

    # Remove empty fields
    cleaned = {k: v for k, v in cleaned.items() if v}

    return cleaned

def main():
    print("UPLOAD CURATED DATASET TO HUGGING FACE")
    print("="*70)

    all_questions = load_all_questions()
    print(f"\nLoaded {len(all_questions)} valid questions")

    topics = {}
    for q in all_questions:
        topic = q['topic']
        topics[topic] = topics.get(topic, 0) + 1

    print("\nDataset statistics:")
    for topic, count in sorted(topics.items()):
        print(f"  {topic}: {count} questions")

    dataset = Dataset.from_list(all_questions)

    hf_username = input("\nEnter your Hugging Face username: ").strip()
    dataset_name = input("Enter dataset name (e.g., logical-reasoning-v5): ").strip()

    if not hf_username or not dataset_name:
        print("\nError: Username and dataset name required")
        return

    repo_id = f"{hf_username}/{dataset_name}"
    private = input(f"\nMake dataset private? (y/n, default=n): ").strip().lower() == 'y'
    confirm = input(f"Proceed with upload to {repo_id}? (yes/no): ").strip().lower()

    if confirm != 'yes':
        print("\nUpload cancelled. Saving local backup...")
        dataset.save_to_disk("curated_dataset_backup")
        with open("curated_dataset_v5.json", "w") as f:
            json.dump(all_questions, f, indent=2, ensure_ascii=False)
        print(f"Saved to: curated_dataset_backup/ and curated_dataset_v5.json")
        return

    try:
        print(f"\nUploading to {repo_id}...")
        dataset.push_to_hub(
            repo_id,
            private=private,
            commit_message="Upload curated logical reasoning dataset v5"
        )
        print(f"\nSUCCESS: https://huggingface.co/datasets/{repo_id}")

        with open("curated_dataset_v5.json", "w") as f:
            json.dump(all_questions, f, indent=2, ensure_ascii=False)
        print(f"Saved backup to: curated_dataset_v5.json")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting: Run 'huggingface-cli login' and check username")

        dataset.save_to_disk("curated_dataset_backup")
        with open("curated_dataset_v5.json", "w") as f:
            json.dump(all_questions, f, indent=2, ensure_ascii=False)
        print(f"Saved backup to: curated_dataset_backup/ and curated_dataset_v5.json")

if __name__ == "__main__":
    main()
