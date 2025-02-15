import json
import re
import os
from datetime import date


def parse_markdown_to_json(md_file_path):
    """
    Parse a Markdown file into a JSON structure containing:
      - metadata (placed first in JSON)
      - categories
      - qa_pairs

    Key features:
    1. 'metadata' is declared first (always).
    2. 'last_updated' is set automatically to today's date.
    3. Parses standard Q&A:
       - We identify categories via lines starting with "## ".
       - We identify Q&A pairs via lines starting with "### Q:".
         The lines after that (until next Q or next category) form the 'answer'.
         Within the answer, we look for:
           - main_points  (default lines or bullet lines if not in example/tips/related topics)
           - examples     (after a line with "📌 **Example**" until next marker)
           - related_topics (after "📌 **Related Topics**")
           - tips         (lines starting with "✅ ")
    4. Also parses "Quick Tips" sections (e.g. "## Quick Tips for Success")
       where there is no "### Q:" but valuable lines to store:
         - We treat them as a single QA pair with a synthetic question, e.g.,
           "What are some Quick Tips for Success?"
         - We parse lines beginning with "✅ " as tips, other bullet lines or text as main_points.
    5. Similar logic for "## Additional Resources & Tips" or any other "##" heading that
       doesn't contain Q: lines. We'll create a single QA pair out of them.

    This ensures that "Quick Tips" blocks or other stand-alone sections also get captured.
    """

    # Main knowledge base structure, with metadata declared first
    knowledge_base = {
        "metadata": {
            "last_updated": str(date.today()),  # auto-set to current date
            "version": "1.0",
            "language": "en",
            "source": "University Knowledge Base",
            "topics_count": 0,
            "qa_pairs_count": 0
        },
        "categories": [],
        "qa_pairs": []
    }

    # Read all lines from the markdown file
    with open(md_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    def generate_id_from_title(title):
        """
        Convert a heading title to a simple category_id by:
         - Lowercasing
         - Replacing non-alphanumerics with underscores
         - Stripping leading/trailing underscores
        """
        cat_id = re.sub(r'[^a-z0-9]+', '_', title.lower().strip())
        return cat_id.strip('_')

    current_category_id = None
    current_category_title = None

    qa_counter = 0  # increments for each QA in the current category

    # For Q&A building
    current_question = None
    current_answer_lines = []

    # We also want to handle the scenario where
    # "## Quick Tips for X" has no "### Q:...", so we create
    # a synthetic question for them. We'll track if we are in that scenario with a flag.
    in_quick_tips_section = False
    quick_tips_section_title = None
    quick_tips_lines = []

    def store_qa_pair():
        """
        If there's a valid question and some collected answer lines, finalize them into
        a QA pair and reset.
        """
        nonlocal current_question, current_answer_lines, qa_counter, current_category_id

        if current_question and current_answer_lines:
            qa_counter += 1
            qa_id = f"{current_category_id}_{qa_counter:03d}"

            parsed_answer = parse_answer_block(current_answer_lines)
            qa_pair = {
                "id": qa_id,
                "category_id": current_category_id,
                "question": current_question.strip(),
                "answer": parsed_answer
            }
            knowledge_base["qa_pairs"].append(qa_pair)

        # reset
        current_question = None
        current_answer_lines = []

    def store_quick_tips_as_qa():
        """
        If we've collected lines for a 'quick tips' or stand-alone section, store them
        as one QA pair with a synthetic question.
        """
        nonlocal quick_tips_section_title, quick_tips_lines, qa_counter, current_category_id

        if quick_tips_section_title and quick_tips_lines:
            # create a synthetic question for the quick tips
            # e.g. "What are some Quick Tips for Physical Wellness?"
            question_text = f"What are some {quick_tips_section_title}?"
            # or you can do a more direct approach if you prefer

            qa_counter += 1
            qa_id = f"{current_category_id}_{qa_counter:03d}"

            parsed_answer = parse_answer_block(quick_tips_lines)

            qa_pair = {
                "id": qa_id,
                "category_id": current_category_id,
                "question": question_text,
                "answer": parsed_answer
            }
            knowledge_base["qa_pairs"].append(qa_pair)

        quick_tips_section_title = None
        quick_tips_lines = []

    def parse_answer_block(answer_lines):
        """
        Parse lines in an answer into:
          - main_points
          - examples
          - tips
          - related_topics

        Markers recognized:
          - "📌 **Example**" -> subsequent lines go to 'examples'
          - "📌 **Related Topics**" -> subsequent lines go to 'related_topics'
          - lines starting with "✅ " -> 'tips'
          - bullet lines ("- " or "* " or "+ ") -> main_points (unless in example/related)
          - everything else goes to main_points by default (unless in example/related)
        """

        main_points = []
        examples = []
        related_topics = []
        tips = []

        collecting_example = False
        collecting_related_topics = False

        for line in answer_lines:
            line_stripped = line.strip()

            # detect example block start
            if line_stripped.lower().startswith("📌 **example"):
                collecting_example = True
                collecting_related_topics = False
                continue
            # detect related topics block
            if line_stripped.lower().startswith("📌 **related topics"):
                collecting_related_topics = True
                collecting_example = False
                continue

            # detect tips
            if line_stripped.startswith("✅ "):
                # everything after '✅ ' is a tip
                tips.append(line_stripped[2:].strip())
                continue

            # bullet lines or plain lines
            # check bullet lines
            bullet_match = re.match(r"^[-*+]\s+(.*)$", line_stripped)
            if bullet_match:
                bullet_text = bullet_match.group(1).strip()
                if collecting_example:
                    examples.append(bullet_text)
                elif collecting_related_topics:
                    related_topics.append(bullet_text)
                else:
                    main_points.append(bullet_text)
            else:
                # plain line
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

    # MAIN LOOP
    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # 1) Check for new category line "## "
        if line.startswith("## "):
            # Before switching categories, store any pending Q/A or quick tips
            store_qa_pair()  # finalize any ongoing Q
            store_quick_tips_as_qa()  # finalize any quick tips block

            # new category
            current_category_title = line.replace("## ", "").strip()
            current_category_id = generate_id_from_title(current_category_title)
            # reset QA counter for new category (if you prefer continuous numbering, remove this)
            qa_counter = 0

            # see if the heading is something like "Quick Tips for X" or "Quick Tips x" or "Additional Resources..."
            # If it starts with "Quick Tips" or "Additional Resources & Tips" => we'll handle it as a block
            # We'll track if we are in quick tips or not.
            # For simplicity let's do a check:
            # If heading contains "Quick Tips" or "Additional Resources & Tips" => we'll handle the lines as a block
            in_quick_tips_section = False
            quick_tips_section_title = None
            quick_tips_lines = []

            # create the category if not exist
            category_exists = any(cat["id"] == current_category_id for cat in knowledge_base["categories"])
            if not category_exists:
                knowledge_base["categories"].append({
                    "id": current_category_id,
                    "title": current_category_title,
                    "subcategories": []
                })

            # If it's something like "Quick Tips for X" or "Quick Tips X" or "Additional Resources & Tips"
            # we handle it as a potential block. We'll do a simple check:
            # - "Quick Tips" in the title or "Additional Resources & Tips" in the title
            if re.search(r"^(quick tips|additional resources)", current_category_title.lower()):
                in_quick_tips_section = True
                quick_tips_section_title = current_category_title  # keep the original
            continue

        # 2) If we detect a question heading "### Q:"
        if line.startswith("### Q:"):
            # finalize any ongoing quick tips block if we were in it
            if in_quick_tips_section:
                store_quick_tips_as_qa()
                in_quick_tips_section = False

            # store the previous question if any
            store_qa_pair()

            current_question = line.replace("### Q:", "").strip()
            current_answer_lines = []
            continue

        # 3) If we are currently in a question, all lines go to "answer"
        if current_question:
            # if line starts with "**A:**"
            if line_stripped.startswith("**A:**"):
                answer_part = line_stripped.replace("**A:**", "").strip()
                if answer_part:
                    current_answer_lines.append(answer_part)
            else:
                current_answer_lines.append(line)
            continue

        # 4) If none of the above, but we are inside a "quick tips" style block (or "additional resources") with no Q lines,
        #    collect the lines. We'll store them as a single QA when we see next "##" or end-of-file
        if in_quick_tips_section:
            # Just collect lines into quick_tips_lines
            # We'll parse them after we detect next category or end
            quick_tips_lines.append(line)

    # End of file => store any final QA or quick tips
    store_qa_pair()
    store_quick_tips_as_qa()

    # update metadata counts
    knowledge_base["metadata"]["topics_count"] = len(knowledge_base["categories"])
    knowledge_base["metadata"]["qa_pairs_count"] = len(knowledge_base["qa_pairs"])

    return knowledge_base


if __name__ == "__main__":
    # Example usage - replace with your actual .md path
    md_file = "/mnt/c/Users/osato/openai_setup/Comprehensive Academic Success Knowledge Base ad875c1ee2404f4ab86aeb15073ae36b-Copy1.md"

    if not os.path.isfile(md_file):
        print(f"Markdown file not found at: {md_file}")
    else:
        data = parse_markdown_to_json(md_file)

        # Save JSON
        output_json_file = "knowledge_base.json"
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"JSON knowledge base created: {output_json_file}")
