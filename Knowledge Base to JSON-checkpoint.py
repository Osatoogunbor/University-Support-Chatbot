#!/usr/bin/env python
# coding: utf-8

# In[9]:


import json
import re
import os
from datetime import date

def parse_markdown_to_json(md_file_path):
    """
    Parse a Markdown file into a JSON structure containing:
      - metadata (shown first in JSON)
      - categories
      - qa_pairs

    Features:
    1. Sets 'metadata' first to ensure it appears first in the final JSON.
    2. Automatically sets 'last_updated' to today's date.
    3. Parses:
       - main_points
       - examples (introduced by "📌 **Example**")
       - related_topics (introduced by "📌 **Related Topics**")
       - tips (lines starting with "✅ ")
    """

    # Knowledge base structure, with metadata declared first
    knowledge_base = {
        "metadata": {
            "last_updated": str(date.today()),   # Update automatically to current date
            "version": "1.0",
            "language": "en",
            "source": "University Knowledge Base",
            "topics_count": 0,
            "qa_pairs_count": 0
        },
        "categories": [],    # Each category is { "id": ..., "title": ..., "subcategories": [] }
        "qa_pairs": []       # Each QA pair is { "id": ..., "category_id": ..., "question": ..., "answer": {...} }
    }

    # Read the entire markdown file
    with open(md_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Helper function to generate a simple category_id from a category title
    def generate_id_from_title(title):
        # Lowercase, remove non-alphanumerics, replace spaces with underscores
        cat_id = re.sub(r'[^a-z0-9]+', '_', title.lower().strip())
        return cat_id.strip('_')  # remove trailing underscores

    current_category_id = None
    current_category_title = None

    # A small counter for QA ID generation
    qa_counter = 0

    # State tracking for Q&A parsing
    current_question = None
    current_answer_lines = []
    
    def store_qa_pair():
        """
        If there's a valid question and some collected answer lines, finalize them into
        a QA pair and reset for the next question.
        """
        nonlocal current_question, current_answer_lines, qa_counter, current_category_id
        
        if current_question and current_answer_lines:
            qa_counter += 1
            qa_id = f"{current_category_id}_{qa_counter:03d}"
            
            # Parse the answer block to structure it into main_points, examples, tips, related_topics
            parsed_answer = parse_answer_block(current_answer_lines)
            
            qa_pair = {
                "id": qa_id,
                "category_id": current_category_id,
                "question": current_question.strip(),
                "answer": parsed_answer
            }
            knowledge_base["qa_pairs"].append(qa_pair)
        
        # Reset
        current_question = None
        current_answer_lines = []

    def parse_answer_block(answer_lines):
        """
        Parse the lines in the answer into:
          - main_points (list)
          - examples (list)
          - tips (list) => lines that start with "✅ "
          - related_topics (list)
        
        If you have more custom rules (e.g., ### Quick Tips sections),
        adapt the logic below accordingly.
        """
        main_points = []
        examples = []
        related_topics = []
        tips = []
        
        # Flags to indicate which section we might be collecting
        collecting_example = False
        collecting_related_topics = False

        for line in answer_lines:
            line_stripped = line.strip()
            
            # --- Special headings or markers ---
            # 1) Examples
            if line_stripped.lower().startswith("📌 **example**"):
                collecting_example = True
                collecting_related_topics = False
                continue
            
            # 2) Related Topics
            if line_stripped.lower().startswith("📌 **related topics**"):
                collecting_example = False
                collecting_related_topics = True
                continue
            
            # 3) Quick Tips style lines: leading "✅ "
            if line_stripped.startswith("✅ "):
                # Everything after "✅ " goes into 'tips'
                tips.append(line_stripped[2:].strip())
                continue
            
            # --- Bullet points and text lines ---
            if re.match(r"^[-*+]\s", line_stripped):
                # It's a bullet line. Decide which list to place it in:
                bullet_text = line_stripped[2:].strip()
                if collecting_example:
                    examples.append(bullet_text)
                elif collecting_related_topics:
                    related_topics.append(bullet_text)
                else:
                    main_points.append(bullet_text)
            else:
                # A plain text line
                if collecting_example:
                    examples.append(line_stripped)
                elif collecting_related_topics:
                    related_topics.append(line_stripped)
                else:
                    main_points.append(line_stripped)
        
        return {
            "main_points": [mp for mp in main_points if mp],
            "examples": [ex for ex in examples if ex],
            "tips": [t for t in tips if t],
            "related_topics": [rt for rt in related_topics if rt]
        }

    # Main parsing loop
    for line in lines:
        # Detect a new category (e.g. "## Time and Task Management")
        if line.startswith("## "):
            # Store any pending QA pair before switching category
            store_qa_pair()
            
            current_category_title = line.replace("## ", "").strip()
            current_category_id = generate_id_from_title(current_category_title)
            
            # Add category if not already present
            category_exists = any(cat["id"] == current_category_id for cat in knowledge_base["categories"])
            if not category_exists:
                knowledge_base["categories"].append({
                    "id": current_category_id,
                    "title": current_category_title,
                    "subcategories": []
                })
            
            continue
        
        # Detect a new question (e.g. "### Q: How do I ...?")
        if line.startswith("### Q:"):
            # Store existing Q/A before starting a new one
            store_qa_pair()
            
            current_question = line.replace("### Q:", "").strip()
            current_answer_lines = []
            continue
        
        # If we're inside a question, everything else is considered part of the answer
        if current_question:
            # Check if line starts with "**A:**"
            if line.strip().startswith("**A:**"):
                answer_part = line.strip().replace("**A:**", "").strip()
                if answer_part:
                    current_answer_lines.append(answer_part)
            else:
                current_answer_lines.append(line)

    # After the loop, store the last Q/A pair (if any)
    store_qa_pair()
    
    # Update metadata counts
    knowledge_base["metadata"]["topics_count"] = len(knowledge_base["categories"])
    knowledge_base["metadata"]["qa_pairs_count"] = len(knowledge_base["qa_pairs"])

    return knowledge_base


if __name__ == "__main__":
    # Your *real* markdown file path below
    md_file = "/mnt/c/Users/osato/openai_setup/Comprehensive Academic Success Knowledge Base ad875c1ee2404f4ab86aeb15073ae36b-Copy1.md"

    
    if not os.path.isfile(md_file):
        print(f"Markdown file not found at: {md_file}")
    else:
        data = parse_markdown_to_json(md_file)
        
        # Save to JSON with metadata placed first
        output_json_file = "knowledge_base.json"
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON knowledge base created: {output_json_file}")


# In[ ]:




