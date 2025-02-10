import os
import re
import json
from datetime import date

def parse_structured_txt_file(filepath):
    """
    Parses a structured Q&A .txt file into a list of Q&A pairs while preserving categories.
    It assumes:
      - '##' indicates a category (e.g., ## Mental Health)
      - '### Q:' indicates a question (e.g., ### Q: What is mental health?)
      - '**A:**' starts the answer section.
    """

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    def generate_id_from_title(title):
        """Generate a normalized ID from the heading text."""
        return re.sub(r'[^a-z0-9]+', '_', title.lower().strip()).strip('_')

    categories = []
    qa_pairs = []

    current_category_id = None
    current_category_title = None
    current_question = None
    current_answer_lines = []
    qa_counter = 0

    def store_qa_and_reset():
        """Save the current Q&A pair to the list and reset accumulators."""
        nonlocal qa_counter, current_question, current_answer_lines

        if current_category_id and current_question and current_answer_lines:
            qa_counter += 1
            qa_id = f"{current_category_id}_{qa_counter:03d}"
            # Save answer as a dictionary even if it’s a plain text answer.
            qa_pairs.append({
                "id": qa_id,
                "category_id": current_category_id,
                "question": current_question.strip(),
                "answer": {
                    "main_points": [" ".join(current_answer_lines).strip()],
                    "examples": [],
                    "tips": [],
                    "related_topics": []
                }
            })

        # Reset variables
        current_question = None
        current_answer_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith("## "):  # New category
            store_qa_and_reset()  # Store previous QA before changing category
            current_category_title = line.replace("##", "").strip()
            current_category_id = generate_id_from_title(current_category_title)

            categories.append({
                "id": current_category_id,
                "title": current_category_title,
                "subcategories": []
            })

        elif line.startswith("### Q:"):  # New question
            store_qa_and_reset()  # Store previous QA before new question
            current_question = line.replace("### Q:", "").strip()

        elif line.startswith("**A:**"):  # Start of answer section
            current_answer_lines.append(line.replace("**A:**", "").strip())

        elif current_question:  # Capture answer lines until the next question or category
            current_answer_lines.append(line)

    # Store the last Q&A pair if any
    store_qa_and_reset()

    return {
        "categories": categories,
        "qa_pairs": qa_pairs
    }

def convert_structured_txt_folder(input_folder, output_json):
    """
    Converts all structured Q&A .txt files into a consolidated JSON file.
    Assumes each .txt file contains structured Q&A with '##' as categories and '### Q:' as questions.
    """

    knowledge_data = {
        "metadata": {
            "last_updated": str(date.today()),
            "version": "1.0",
            "language": "en",
            "source": "Extracted Documents (Structured)",
            "topics_count": 0,
            "qa_pairs_count": 0
        },
        "categories": [],
        "qa_pairs": []
    }

    existing_cat_ids = set()
    file_count = 0

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(".txt"):
            continue

        file_count += 1
        file_path = os.path.join(input_folder, fname)

        single_doc_data = parse_structured_txt_file(file_path)

        # Merge categories
        for cat in single_doc_data["categories"]:
            if cat["id"] not in existing_cat_ids:
                knowledge_data["categories"].append(cat)
                existing_cat_ids.add(cat["id"])

        # Merge QA pairs
        knowledge_data["qa_pairs"].extend(single_doc_data["qa_pairs"])

    # Update metadata
    knowledge_data["metadata"]["topics_count"] = len(knowledge_data["categories"])
    knowledge_data["metadata"]["qa_pairs_count"] = len(knowledge_data["qa_pairs"])

    # Save as JSON
    with open(output_json, "w", encoding="utf-8") as out_f:
        json.dump(knowledge_data, out_f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully processed {file_count} .txt files.")
    print(f"   Categories: {knowledge_data['metadata']['topics_count']}")
    print(f"   QA Pairs:  {knowledge_data['metadata']['qa_pairs_count']}")
    print(f"   Output => {output_json}")

if __name__ == "__main__":
    input_folder = "/mnt/c/Users/osato/Downloads/extracted_txt"
    output_file ="/mnt/c/Users/osato/openai_setup/extracted_structured.json"
    convert_structured_txt_folder(input_folder, output_file)
