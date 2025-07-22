import argparse
import os
import pickle
import warnings

import pandas as pd
import rootutils
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from utils import flatten, load_data, preprocess_and_parse_output

# Setup
warnings.filterwarnings("ignore")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def build_prompt(sentence, model_name):
    prompt = """You are an information extraction system. Your task is to perform Named Entity Recognition (NER) on the sentence below.

    Extract all named entities from the sentence and return each as a **separate, valid JSON object**.

    Allowed entity types:
    - politician
    - person
    - organization
    - politicalparty
    - event
    - election
    - country
    - location
    - miscellaneous

    Instructions:
    - For each entity, return a JSON object with exactly two fields:
      - "sentence": the exact text span from the input sentence representing the entity.
      - "type": one of the allowed entity types listed above.
    - If no entity is found, return a single JSON object with both fields set to empty strings:
      {"sentence": "", "type": ""}
    - Do **not** wrap the JSON objects in a list (no square brackets).
    - Separate multiple JSON objects using **commas and a single space only**, like this:
      {"sentence": "...", "type": "..."}, {"sentence": "...", "type": "..."}
    - The output must be strictly valid JSON:
      - Use double quotes only
      - Close all braces properly
      - Do not include trailing commas
    - Do not include any explanation, notes, or extra text. Output **only** the JSON objects.

    Sentence:
    "%s" """
    prompt_ = prompt % sentence
    if model_name == "Mistral-7B-Instruct-v0.3" or model_name == "Llama-3.1-8B-Instruct":
        return f"[INST] {prompt_} [/INST]"
    elif model_name == "gemma-3-4b-it":
        return (f"<start_of_turn>user "
                f"{prompt_} <end_of_turn>"
                f"<start_of_turn>model")


def main(model_id: str, dataset: pd.DataFrame, batch_size: int = 8):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    generator = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        task="text-generation",
        do_sample=False,
        max_new_tokens=200,
        repetition_penalty=1.1,
        torch_dtype=torch.bfloat16
    )

    prompts = dataset["prompt"].tolist()
    sentences = dataset["sentence"].tolist()
    gold_labels = dataset["ner_tag"].apply(lambda x: x.split() if isinstance(x, str) else []).tolist()
    assert len(sentences) == len(gold_labels)

    preds = []
    labels_cleaned = []
    errors = 0
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_sentences = sentences[i:i + batch_size]
        batch_labels = gold_labels[i:i + batch_size]

        with torch.inference_mode():
            outputs = generator(batch_prompts)

        for j, output_dict in enumerate(outputs):
            row_sentence = batch_sentences[j].lower().strip()
            original_tokens = row_sentence.split()
            ner_mask = ['o'] * len(batch_labels[j])
            output = output_dict[0]["generated_text"]

            try:
                # output = output.replace("\\", "").replace("\n", "")
                # if "[/INST]" in output:
                #     output = re.sub(r'\s*\[/INST\].*', '', output, flags=re.DOTALL)
                # output = re.sub(r'^.*?{', '{', output)
                # output = re.sub(r'}[^}]*$', '}', output)
                # output = re.sub(r'}\s*{', '}, {', output)
                # output = remove_extra_closing_braces(output)
                # output = f"[{output}]"
                # parsed_data = json.loads(output)
                parsed_data = preprocess_and_parse_output(output)

                for item in parsed_data:
                    pred_sentence = item.get('sentence', '').lower().strip().split()
                    pred_type = item.get('type', '').lower().strip()
                    if not pred_sentence or not pred_type:
                        continue
                    if len(pred_sentence) > len(original_tokens):
                        continue
                    if " ".join(pred_sentence) not in row_sentence:
                        continue
                    first_token = pred_sentence[0]
                    try:
                        idx = original_tokens.index(first_token)
                        ner_mask[idx] = f"b-{pred_type}"
                        for k in range(1, len(pred_sentence)):
                            ner_mask[idx + k] = f"i-{pred_type}"
                    except ValueError:
                        continue

                preds.append(ner_mask[:len(batch_labels[j])])
                labels_cleaned.append(batch_labels[j])

            except Exception as e:
                print(f"[Batch {i}] Generator error: {e}")
                print(f"Output: {output_dict[0]['generated_text']} \n \n")
                errors += 1
                continue

    preds = flatten(preds)
    labels_cleaned = flatten(labels_cleaned)

    #  todo: testing
    if len(preds) != len(labels_cleaned):
        with open(f'preds_ner_{model_id.split("/")[-1]}.pkl', 'w') as f:
            pickle.dump([preds], f)
        with open(f'labels_cleaned_ner_{model_id.split("/")[-1]}.pkl', 'w') as f:
            pickle.dump([labels_cleaned], f)

    assert len(preds) == len(labels_cleaned), "Mismatch between predictions and gold labels"

    return {
        'accuracy': accuracy_score(y_true=labels_cleaned, y_pred=preds),
        'precision': precision_score(y_true=labels_cleaned, y_pred=preds, average="macro"),
        'recall': recall_score(y_true=labels_cleaned, y_pred=preds, average="macro"),
        'f1': f1_score(y_true=labels_cleaned, y_pred=preds, average="macro"),
        'errors': errors
    }


if __name__ == "__main__":
    # model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # model_id = "google/gemma-3-4b-it"
    # model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier from Hugging Face")
    args = parser.parse_args()

    model_id = args.model_id
    model_name = model_id.split("/")[-1]

    dataset = load_data("data/ner/test.json")
    dataset["prompt"] = dataset["sentence"].apply(lambda x: build_prompt(x, model_name=model_name))

    results = []
    models_results = main("/lustre/fsmisc/dataset/HuggingFace_Models/" + model_id, dataset)

    results.append({
        'model': model_name,
        'accuracy': models_results['accuracy'],
        'precision': models_results['precision'],
        'recall': models_results['recall'],
        'f1': models_results['f1'],
        'errors': models_results['errors']
    })

    print("################################### RESULTS ###################################")
    print("Model:", model_name)
    print(results)
    print("###############################################################################")

    with pd.ExcelWriter(os.path.join(rootutils.find_root(""), f"results_{model_name}.xlsx")) as writer:
        df = pd.DataFrame(results)
        df.to_excel(writer, sheet_name="ner", index=False)
