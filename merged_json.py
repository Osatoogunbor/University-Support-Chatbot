#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re

def load_json_safely(file_path):
    """Load JSON from file_path safely, returning a dict or None if failure."""
    if not os.path.isfile(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Could not decode JSON from {file_path}: {e}")
        return None

def add_emergency_flag(qa_pairs, emergency_keywords):
    """
    For each QA pair in qa_pairs:
      - Ensure its "answer" is stored as a dictionary with keys:
        "main_points", "examples", "tips", and "related_topics".
      - Check the first element of "main_points" (converted to lowercase)
        for any emergency keywords.
      - If a keyword is found, add the flag "is_emergency": true.
    """
    for qa in qa_pairs:
        # If the answer is a plain string, wrap it in a dictionary.
        if isinstance(qa.get("answer"), str):
            qa["answer"] = {
                "main_points": [qa["answer"].strip()],
                "examples": [],
                "tips": [],
                "related_topics": []
            }
        # Get the main text from the first element of main_points (if any)
        main_text = qa["answer"]["main_points"][0].lower() if qa["answer"]["main_points"] else ""
        if any(keyword in main_text for keyword in emergency_keywords):
            qa["is_emergency"] = True

def merge_knowledge_bases(existing_json_path, new_extracted_json_path, merged_json_path):
    # 1. Load existing data
    existing_data = load_json_safely(existing_json_path)
    if existing_data is None:
        print("❌ Aborting: missing or invalid existing knowledge base JSON.")
        return

    # 2. Load new extracted data
    new_data = load_json_safely(new_extracted_json_path)
    if new_data is None:
        print("❌ Aborting: missing or invalid extracted QA JSON.")
        return

    # Define keywords that indicate generic emergency advice.
    emergency_keywords = ["i need help now", "emergency", "urgent", "crisis"]

    # Process existing QA pairs:
    print("🔍 Checking existing QA pairs for emergency keywords...")
    for qa in existing_data.get("qa_pairs", []):
        # Tag existing QA pairs if not already tagged.
        qa.setdefault("source", "existing")
    add_emergency_flag(existing_data.get("qa_pairs", []), emergency_keywords)

    # Process new QA pairs for emergency flag.
    print("🔍 Checking new QA pairs for emergency keywords...")
    for qa in new_data.get("qa_pairs", []):
        print(f"QA question: {qa.get('question')}, answer type: {type(qa.get('answer'))}")
        # Tag new (extracted) QA pairs.
        qa["source"] = "extracted"
    add_emergency_flag(new_data.get("qa_pairs", []), emergency_keywords)

    # 3. Check existing_data structure
    if "categories" not in existing_data or "qa_pairs" not in existing_data:
        print("❌ Error: 'categories' or 'qa_pairs' missing in existing knowledge base.")
        return

    # 4. Check new_data structure
    if "categories" not in new_data or "qa_pairs" not in new_data:
        print("❌ Error: 'categories' or 'qa_pairs' missing in extracted data.")
        return

    # 5. Merge categories (append only those that are new)
    existing_cat_ids = {cat["id"] for cat in existing_data["categories"]}
    for cat in new_data["categories"]:
        cat_id = cat.get("id")
        if cat_id not in existing_cat_ids:
            existing_data["categories"].append(cat)
            existing_cat_ids.add(cat_id)
        else:
            # Optionally merge subcategories here if needed.
            pass

    # 6. Merge QA pairs (append new ones to existing ones)
    new_qa_pairs = new_data.get("qa_pairs", [])
    existing_data["qa_pairs"].extend(new_qa_pairs)

    # 7. Update metadata
    existing_data["metadata"]["topics_count"] = len(existing_data["categories"])
    existing_data["metadata"]["qa_pairs_count"] = len(existing_data["qa_pairs"])

    # 8. Save merged JSON
    try:
        with open(merged_json_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Merged knowledge base saved to: {merged_json_path}")
    except Exception as e:
        print(f"❌ Error saving merged knowledge base: {e}")

if __name__ == "__main__":
    # Example usage: update with your actual file paths
    base_dir = "/mnt/c/Users/osato/openai_setup"
    existing_kb_path = os.path.join(base_dir, "knowledge_base.json")
    extracted_kb_path = os.path.join(base_dir, "extracted_structured.json")
    merged_output_path = os.path.join(base_dir, "merged_knowledge_base.json")

    merge_knowledge_bases(
        existing_json_path=existing_kb_path,
        new_extracted_json_path=extracted_kb_path,
        merged_json_path=merged_output_path
    )

