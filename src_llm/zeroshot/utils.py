import json
import re

import pandas as pd


def flatten(xss):
    return [x for xs in xss for x in xs]


def load_data(filepath):
    with open(filepath, 'r') as f:
        raw_data = json.load(f)
    records = []
    for entry in raw_data:
        sentence = " ".join(entry['tokens']).strip()
        ner_tag = " ".join(entry['ner_tags']).strip().lower()
        records.append({
            "sentence": sentence,
            "ner_tag": ner_tag
        })
    return pd.DataFrame(records)


def remove_extra_closing_braces(s):
    stack = []
    result = []
    matching = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in matching.values():
            stack.append(char)
            result.append(char)
        elif char in matching:
            if stack and stack[-1] == matching[char]:
                stack.pop()
                result.append(char)
            else:
                # skip this extra closing brace
                continue
        else:
            result.append(char)

    return ''.join(result)


def preprocess_and_parse_output(output: str):
    # Basic cleaning
    output = output.replace("\\", "").replace("\n", "").strip()

    # Remove [/INST] and everything after
    if "[/INST]" in output:
        output = re.sub(r'\s*\[/INST\].*', '', output, flags=re.DOTALL)

    # Extract everything between the first { and the last }
    output = re.sub(r'^.*?{', '{', output)
    output = re.sub(r'}[^}]*$', '}', output)

    # Replace everything that is between brackets with comma
    output = re.sub(r'}[^{}]*{', '},{', output)
    output = re.sub(r'}\s*{', '}, {', output)

    # Fix malformed stringified list of JSON: ["{...}", "{...}"]
    if re.match(r'^\[\s*"{.*}"\s*(,\s*"{.*}"\s*)*\]$', output):
        try:
            string_list = json.loads(output)
            output = "[" + ", ".join(string_list) + "]"
        except Exception:
            pass  # fallback to next step if still broken

    # Fix malformed: ["{...}"] with one unescaped JSON string
    if re.match(r'^\s*\["\s*\{.*\}\s*"\]\s*$', output):
        inner = re.sub(r'^\s*\["\s*|\s*"\]\s*$', '', output)
        output = f"[{inner}]"

    # Remove unmatched extra closing braces (e.g., {"a":"b"}})
    output = remove_extra_closing_braces(output)

    output = output.encode('utf-8').decode('unicode_escape')

    # Ensure it is a list
    if not output.strip().startswith('['):
        output = f"[{output}]"

    # Final parse attempt
    parsed_data = json.loads(output)

    # If it's a list of JSON strings, parse them
    if isinstance(parsed_data, list) and all(isinstance(item, str) for item in parsed_data):
        parsed_data = [json.loads(item) for item in parsed_data]

    elif isinstance(parsed_data, dict):
        parsed_data = [parsed_data]

    return parsed_data
