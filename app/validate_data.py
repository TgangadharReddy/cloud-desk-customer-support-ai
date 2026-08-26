import json
from collections import Counter
from pathlib import Path


REQUIRED_FIELDS = {"id", "category", "question", "answer"}
VALID_CATEGORIES = {"Billing", "Technical", "Account Access"}


def validate_faq_data():
    faq_path = Path("data/faq.json")

    print("=" * 50)
    print("CloudDesk FAQ Data Validation")
    print("=" * 50)

    # Load JSON
    try:
        with open(faq_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"ERROR: FAQ file not found: {faq_path}")
        return False
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON: {exc}")
        return False

    if not isinstance(data, list):
        print("ERROR: FAQ data must be a JSON list.")
        return False

    print(f"Total records: {len(data)}")

    errors = []

    # Required fields
    missing_fields = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Record {index + 1} is not an object.")
            continue

        missing = REQUIRED_FIELDS - set(item.keys())

        if missing:
            missing_fields.append(
                f"Record {index + 1} ({item.get('id', 'unknown')}): "
                f"missing {sorted(missing)}"
            )

    print(f"Missing required fields: {len(missing_fields)}")

    for error in missing_fields:
        errors.append(error)

    # Empty questions / answers
    empty_questions = []
    empty_answers = []

    for item in data:
        question = item.get("question")
        answer = item.get("answer")

        if not isinstance(question, str) or not question.strip():
            empty_questions.append(item.get("id", "unknown"))

        if not isinstance(answer, str) or not answer.strip():
            empty_answers.append(item.get("id", "unknown"))

    print(f"Empty questions: {len(empty_questions)}")
    print(f"Empty answers: {len(empty_answers)}")

    if empty_questions:
        errors.append(
            f"Empty questions: {', '.join(empty_questions)}"
        )

    if empty_answers:
        errors.append(
            f"Empty answers: {', '.join(empty_answers)}"
        )

    # Duplicate IDs
    ids = [item.get("id") for item in data if isinstance(item, dict)]
    id_counts = Counter(ids)

    duplicate_ids = sorted(
        item_id for item_id, count in id_counts.items()
        if item_id is not None and count > 1
    )

    print(f"Duplicate IDs: {len(duplicate_ids)}")

    if duplicate_ids:
        errors.append(
            f"Duplicate IDs: {', '.join(duplicate_ids)}"
        )

    # Duplicate questions
    questions = [
        item.get("question", "").strip().lower()
        for item in data
        if isinstance(item, dict)
    ]

    question_counts = Counter(questions)

    duplicate_questions = sorted(
        question for question, count in question_counts.items()
        if question and count > 1
    )

    print(f"Duplicate questions: {len(duplicate_questions)}")

    if duplicate_questions:
        errors.append(
            f"Duplicate questions found: {len(duplicate_questions)}"
        )

    # Category validation
    categories = [
        item.get("category")
        for item in data
        if isinstance(item, dict)
    ]

    invalid_categories = sorted(
        category
        for category in set(categories)
        if category not in VALID_CATEGORIES
    )

    category_counts = Counter(categories)

    print("\nCategory distribution:")
    for category in sorted(VALID_CATEGORIES):
        print(f"  {category}: {category_counts.get(category, 0)}")

    print(f"Invalid categories: {len(invalid_categories)}")

    if invalid_categories:
        errors.append(
            f"Invalid categories: {', '.join(map(str, invalid_categories))}"
        )

    # Final result
    print("\n" + "=" * 50)

    if errors:
        print("STATUS: FAIL")
        print("=" * 50)

        print("\nProblems found:")
        for error in errors:
            print(f"- {error}")

        return False

    print("STATUS: PASS")
    print("=" * 50)

    return True


if __name__ == "__main__":
    success = validate_faq_data()
    raise SystemExit(0 if success else 1)