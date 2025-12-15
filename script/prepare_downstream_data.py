"""
Run the download.sh script before this.
This file will preprocess all the datasets needed for the downstream tasks.
"""
import ast
import json
import logging
import os
import random
import re
import shutil

import nltk
import pandas as pd
import rootutils
import spacy
from sklearn.model_selection import train_test_split
from unidecode import unidecode

nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_text(text):
    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Collapse multiple spaces to a single space
    text = re.sub(r"\s{2,}", " ", text)
    # Strip leading/trailing spaces
    text = text.strip()
    return text


def adjust_conll_data(df: pd.DataFrame):
    """
    Adjust and clean CONLL data to ensure the token and NER tag lengths match.
    This handles edge cases where the tag list may be longer than the token list.
    """
    data = {"tokens": [], "ner_tags": []}
    for token, tag in zip(df['tokens'], df['ner_tags']):
        tag = tag[len(tag) - len(token):] if len(token) < len(tag) else tag
        data["tokens"].append(token)
        data["ner_tags"].append(tag)
    df = pd.DataFrame(data).dropna().reset_index(drop=True)
    return df


def save_conll_data(df: pd.DataFrame, file_path: str):
    """
    Save tokenized CONLL-style data as a JSON file.
    Each sentence is stored with an ID, tokens, and corresponding NER tags.
    """
    json_data = []
    for i, row in df.iterrows():
        json_obj = {
            "id": str(i),
            "ner_tags": row['ner_tags'],
            "tokens": row['tokens']
        }
        json_data.append(json_obj)

    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    logging.info(f"Saved CONLL data to {file_path}")


def preprocess_elecdeb60to20_components(folder):
    files = os.listdir(folder)

    for f in files:
        data = []
        file_path = os.path.join(folder, f)
        current_sentence = {"tokens": [], "ner_tags": []}

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.rstrip("\n")

                # NEW: blank line triggers new speech turn
                if line == "":
                    # close current turn if it has content
                    if current_sentence["tokens"]:
                        assert len(current_sentence["tokens"]) == len(current_sentence["ner_tags"])
                        data.append(current_sentence)

                    # reset and start skipping first two speaker lines
                    current_sentence = {"tokens": [], "ner_tags": []}
                    continue

                # Normal token/tag line
                parts = line.split("\t")
                token, tag = parts[0], parts[-1]
                token = unidecode(token)

                current_sentence["tokens"].append(token)
                current_sentence["ner_tags"].append(tag)

            # Add last sentence if not empty
            if current_sentence["tokens"]:
                assert len(current_sentence["tokens"]) == len(current_sentence["ner_tags"])
                data.append(current_sentence)

        # Remove original file
        os.remove(file_path)
        logging.info(f"Deleted CONLL file: {file_path}")

        # Shuffle + split
        df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
        save_conll_data(df, os.path.join(folder, f"{f.split('.')[0]}.json"))


def preprocess_elecdeb60to20_relations(folder):
    """
    The authors frame this task as multi-class classification problem.
    """
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    id2labels = {0: 'attack', 1: 'support', 2: 'no_relation'}
    labels = None
    for file in files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path)
        os.remove(file_path)
        df.rename({'relation_type': 'label'}, axis='columns', inplace=True)
        if labels is None:
            labels = {key: idx for idx, key in enumerate(sorted(set(df['label'].tolist())))}
        df['text'] = df['subject'] + " [SEP] " + df['object']
        df['label'] = df['label'].map(labels)
        df = df[['text', 'label']].dropna().drop_duplicates().reset_index(drop=True)
        df['label_value'] = df['label'].map(id2labels)
        df.to_csv(file_path, index=False)


def preprocess_parl_vote(folder):
    """
    The authors of ParlVote perform sentiment analysis on UK Parliamentary Debates. They achieve the best result using
    Bert + MLP with a concatenation of Motion+Speech text (Accuracy - 67.31)
    """
    file_path = os.path.join(folder, os.listdir(folder)[0])
    df = pd.read_csv(file_path)[['speech', 'motion_text', 'vote']]
    df['text'] = df['motion_text'] + " " + df['speech']
    df['text'] = df['text'].apply(lambda s: clean_text(s))
    df.rename({"vote": "label"}, inplace=True, axis=1)
    df.drop(['speech', 'motion_text'], axis=1, inplace=True)
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # Split the dataset
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    test, dev = train_test_split(test, test_size=0.5, random_state=42)

    # Save to CSV
    train.to_csv(os.path.join(folder, "train.csv"), index=False)
    test.to_csv(os.path.join(folder, "test.csv"), index=False)
    dev.to_csv(os.path.join(folder, "dev.csv"), index=False)

    # Remove original file
    os.remove(file_path)


def process_aus_hansard(folder: str):
    """
    The authors of the paper *Stance Classification: A Comparative Study and Use Case on Australian Parliamentary Debates*
    annotated a portion of the Australian Hansard for stance detection, using the speakers' votes as labels.
    Their goal was to frame this as a cross-domain challenge: models were trained on augmented versions of
    ParlVote + HanDeSet, and then evaluated on the Australian Hansard dataset they collected. The evaluation
    metrics included Accuracy, F1 score, and AUROC.
    """
    parlvote_df = pd.read_csv(os.path.join(folder, "parlvote_features.csv"))[['speech', 'label']]
    parlvote_df.rename({'speech': 'text'}, inplace=True, axis=1)

    handeset_df = pd.read_csv(os.path.join(folder, "handeset_features.csv"))[['speech', 'label']]
    handeset_df.rename({'speech': 'text'}, inplace=True, axis=1)

    aus_df = pd.read_csv(os.path.join(folder, "aus_augmented_features.csv"))[['hypothesis', 'label']]
    aus_df.rename({'hypothesis': 'text'}, inplace=True, axis=1)
    aus_df = aus_df.dropna().drop_duplicates().reset_index(drop=True)

    full_df = pd.concat([parlvote_df, handeset_df], axis=0).dropna().drop_duplicates().reset_index(drop=True)
    train, dev = train_test_split(full_df, test_size=0.2, random_state=42)

    train.to_csv(os.path.join(folder, "train.csv"), index=False)
    dev.to_csv(os.path.join(folder, "dev.csv"), index=False)
    aus_df.to_csv(os.path.join(folder, "test.csv"), index=False)

    os.remove(os.path.join(folder, "parlvote_features.csv"))
    os.remove(os.path.join(folder, "handeset_features.csv"))
    os.remove(os.path.join(folder, "aus_augmented_features.csv"))


def process_con_vote(folder: str):
    """
    The authors of the paper *Get out the vote: Determining support or opposition from congressional floor-debate transcripts.*
    annotated a dataset of Us Congressional floor debates for stance classification purposes using the vote of the politician.
    Their dataset is annotated at the speech level. In our work, we only use the speeches, not the relations.
    Bets result with only speeches: Acc 0.6605
    """
    folders = {
        'train': 'training_set',
        'dev': 'development_set',
        'test': 'test_set'
    }
    label2id = None
    for key, value in folders.items():
        dataset = []
        files = os.listdir(os.path.join(folder, "convote_v1.1/data_stage_three", value))
        for file in files:
            with open(os.path.join(folder, "convote_v1.1/data_stage_three", value, file), 'r') as f:
                text = f.read()
            label = file.split("_")[-1][2]
            dataset.append({
                'text': text,
                'label': label
            })
        df = pd.DataFrame(dataset).dropna().drop_duplicates().reset_index(drop=True)
        assert len(df['label'].unique().tolist()) == 2
        if label2id is None:
            label2id = {key: idx for idx, key in enumerate(sorted(set(df['label'].tolist())))}
        df['label'] = df['label'].map(label2id)
        df['text'] = df['text'].apply(clean_text)
        df.to_csv(os.path.join(folder, f"{key}.csv"), index=False)

    shutil.rmtree(os.path.join(folder, "convote_v1.1"))


def process_han_de_set(folder: str):
    """
    The authors of the paper *'Aye' or 'No'? Speech-level Sentiment Analysis of Hansard UK Parliamentary Debate Transcripts*
    classify the sentiment polarity of politicians towards motions (positive/negative) using their votes.
    They classify the sentiment of both speeches and motions. They do it at the speech level.
    They use a 2-step model but in our work we will use only the speeches in a one-step manner.
    Best performance with 1-step model: 0.699 (vote labels) and 0.713 (manual labels).
    """
    df = pd.read_csv(os.path.join(folder, "file_downloaded"))[['manual speech', 'utt1', 'utt2', 'utt3', 'utt4', 'utt5']]

    dataset = []
    for _, row in df.iterrows():
        utt = []
        for i in range(1, 6):
            text = str(row['utt' + str(i)])
            if text is not None and len(str(text)) > 0:
                utt.append(text)
        dataset.append({
            'text': clean_text(' '.join(utt)),
            'label': row['manual speech'],
        })

    df = pd.DataFrame(dataset).dropna().drop_duplicates().reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    test, dev = train_test_split(test, test_size=0.5, random_state=42)

    train.to_csv(os.path.join(folder, "train.csv"), index=False)
    dev.to_csv(os.path.join(folder, "dev.csv"), index=False)
    test.to_csv(os.path.join(folder, "test.csv"), index=False)

    os.remove(os.path.join(folder, "file_downloaded"))


def preprocess_parl_vote_plus(folder: str):
    """
    The authors classify the policy of the parliament members towards speeches (positive/negative) using annotated dataset.
    The task is multi-classification. BWe can also do multi-label classification by classifying both sentiment and policy preferences.
    """
    df = pd.read_csv(os.path.join(folder, "ParlVote2_1.csv"))
    df = df[['speech', 'policy_preference', 'vote']]
    df.rename({'policy_preference': 'label', 'speech': 'text'}, axis=1, inplace=True)
    labels2id = {key: idx for idx, key in enumerate(sorted(set(df['label'].tolist())))}
    id2_labels_value = {104.0: 'Military: Positive',
                        105.0: 'Military: Negative',
                        106.0: 'Peace',
                        108.0: 'European Union: Positive',
                        110.0: 'European Union: Negative',
                        201.2: 'Human Rights',
                        202.4: 'Direct Democracy: Positive',
                        203.0: 'Constitutionalism: Positive',
                        204.0: 'Constitutionalism: Negative',
                        301.0: 'Decentralisation: Positive',
                        302.0: 'Centralisation: Positive',
                        304.0: 'Political Corruption',
                        305.1: 'Political Authority: Party',
                        305.2: 'Political Authority: Personal',
                        401.0: 'Free Market Economy',
                        402.0: 'Incentives: Positive',
                        403.0: 'Market Regulation',
                        411.0: 'Technology: Positive',
                        413.0: 'Nationalisation',
                        501.0: 'Environmental Protection',
                        503.0: 'Equality: Positive',
                        504.0: 'Welfare State Expansion',
                        505.0: 'Welfare State Limitation',
                        506.0: 'Education Expansion',
                        507.0: 'Education Limitation',
                        601.2: 'Immigration: Negative',
                        602.2: 'Immigration: Positive',
                        603.0: 'Traditional Morality: Positive',
                        604.0: 'Traditional Morality: Negative',
                        605.1: 'Law and Order: Positive',
                        605.2: 'Law and Order: Negative',
                        701.0: 'Labour Groups: Positive',
                        702.0: 'Labour Groups: Negative',
                        706.0: 'Underprivileged Minority Groups'
                        }

    df['label_value'] = df['label'].map(id2_labels_value)
    df['label'] = df['label'].map(labels2id)
    df['multi_label'] = df.apply(lambda row: f"policy:{row['label']};vote:{row['vote']}", axis=1)
    df[['text', 'label', 'label_value', 'multi_label']].dropna().drop_duplicates().reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    dev, test = train_test_split(test, test_size=0.5, random_state=42)

    train.to_csv(os.path.join(folder, "train.csv"), index=False)
    dev.to_csv(os.path.join(folder, "dev.csv"), index=False)
    test.to_csv(os.path.join(folder, "test.csv"), index=False)

    os.remove(os.path.join(folder, "ParlVote2_1.csv"))


def preprocess_motion_policy_preferences(folder: str):
    df = pd.read_csv(os.path.join(folder, "MotionPolicyPreferences - Gold.csv"),
                     names=['quasi-sentence ID', 'debate title', 'motion text',
                            'quasi-sentence policy preference code label', 'motion policy preference code label'])
    df.rename({'motion text': 'text'}, axis=1, inplace=True)
    labels2id = {key: idx for idx, key in
                 enumerate(sorted(set(df['quasi-sentence policy preference code label'].tolist())))}
    id2_labels_value = {104: 'Military: Positive',
                        105: 'Military: Negative',
                        106: 'Peace',
                        107: 'Internationalism: Positive',
                        108: 'European Union: Positive',
                        110: 'European Union: Negative',
                        201: 'Human Rights',
                        202: 'Direct Democracy: Positive',
                        203: 'Constitutionalism: Positive',
                        204: 'Constitutionalism: Negative',
                        301: 'Decentralisation: Positive',
                        302: 'Centralisation: Positive',
                        303: 'Governative and Administrative Efficiency',
                        304: 'Political Corruption',
                        305: 'Political Authority: Party',
                        401: 'Free Market Economy',
                        402: 'Incentives: Positive',
                        403: 'Market Regulation',
                        404: 'Economic Planning',
                        405: 'Corporatism/Mixed Economy',
                        407: 'Protectionism: Negative',
                        408: 'Economic Goals',
                        409: 'Keynesian Demand Management',
                        410: 'Economic Growth: Positive',
                        411: 'Technology and Infrastructure: Positive',
                        412: 'Controlled Economy',
                        413: 'Nationalisation',
                        414: 'Economic Orthodoxy',
                        501: 'Environmental Protection',
                        502: 'Culture: Positive',
                        503: 'Equality: Positive',
                        504: 'Welfare State Expansion',
                        505: 'Welfare State Limitation',
                        506: 'Education Expansion',
                        601: 'National Way of Life: Positive',
                        602: 'National Way of Life: Negative',
                        605: 'Law and Order: Positive',
                        606: 'Civic Mindedness: Positive',
                        607: 'Multiculturalism: Positive',
                        608: 'Multiculturalism: Negative',
                        701: 'Labour Groups: Positive',
                        703: 'Agriculture and Farmers: Positive',
                        704: 'Middle Class and Professional Groups',
                        705: 'Underprivileged Minority Groups',
                        706: 'Non-economic Demographic Groups'
                        }
    df['label_value'] = df['quasi-sentence policy preference code label'].map(id2_labels_value)
    df['label'] = df['quasi-sentence policy preference code label'].map(labels2id)
    df.drop([col for col in df.columns if col not in ['label', 'label_value', 'text']], axis=1, inplace=True)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    test, dev = train_test_split(test, test_size=0.5, random_state=42)
    train.to_csv(os.path.join(folder, "train.csv"), index=False)
    dev.to_csv(os.path.join(folder, "dev.csv"), index=False)
    test.to_csv(os.path.join(folder, "test.csv"), index=False)
    os.remove(os.path.join(folder, "MotionPolicyPreferences - Gold.csv"))


def preprocess_ArgUNSC(folder: str):
    df = pd.read_csv(os.path.join(folder % "sequence_labelling", "base.csv"))

    def process_argument_detection(df: pd.DataFrame):
        df_components = df[['Full_Sentence', 'Component', 'Component_Type']]
        dataset = []
        malformed = 0
        for row_idx, row in df_components.iterrows():
            speech = row['Full_Sentence']
            speech = speech.strip().replace("\r", "").replace("\n", " ")
            try:
                component = ast.literal_eval(row['Component'])[-1]
                component = component.strip().replace("\r", "").replace("\n", " ")
            except:
                malformed += 1
                continue
            label = row['Component_Type']

            doc = nlp(speech)
            tokens = [tok.text for tok in doc]

            # Initialize O-tags
            ner_tags = ["O"] * len(tokens)

            if label == "non-arg":
                # Non argumentative component
                dataset.append({"tokens": tokens, "ner_tags": ner_tags})
                continue

            # Find the character span of the component inside the sentence
            comp_start = speech.find(component)
            if comp_start == -1:
                # Component not found → leave all O-tags
                dataset.append({"tokens": tokens, "ner_tags": ner_tags})
                continue

            comp_end = comp_start + len(component)

            # Assign BIO tags based on token offsets
            for i, tok in enumerate(doc):
                tok_start, tok_end = tok.idx, tok.idx + len(tok)

                # Check overlap between token span and component span
                if tok_end <= comp_start or tok_start >= comp_end:
                    continue  # no overlap

                if not any(tag.endswith(label) for tag in ner_tags):
                    ner_tags[i] = f"B-{label}"
                else:
                    ner_tags[i] = f"I-{label}"

            dataset.append({"id": row_idx, "tokens": tokens, "ner_tags": ner_tags})
        new_df = pd.DataFrame(dataset).reset_index(drop=True)
        train, test = train_test_split(new_df, test_size=0.2, random_state=42)
        dev, test = train_test_split(test, test_size=0.5, random_state=42)

        save_conll_data(train, os.path.join(folder % "sequence_labelling", "train.json"))
        save_conll_data(dev, os.path.join(folder % "sequence_labelling", "dev.json"))
        save_conll_data(test, os.path.join(folder % "sequence_labelling", "test.json"))

    def relation_prediction(df_original):
        df = df_original[(df_original["Component"] != "non-arg") & (df_original['Premises'] != "standalone")].copy()
        df["Component"] = df["Component"].apply(ast.literal_eval)
        df["Premises"] = df["Premises"].apply(lambda x: ast.literal_eval(x))

        df_components = (
            df["Component"]
            .apply(lambda x: {"id": x[0], "component": x[-1]})
            .apply(pd.Series)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        comp_ids = set(df_components["id"])

        dataset_pos = []
        for comp_row, prem_row in zip(df["Component"], df["Premises"]):
            comp_id = comp_row[0]

            for key, value in prem_row.items():
                if key in comp_ids:
                    dataset_pos.append((comp_id, key, value))

        dataset_pos = pd.DataFrame(dataset_pos, columns=["comp1", "comp2", "label"])
        pos_pairs = set(zip(dataset_pos["comp1"], dataset_pos["comp2"]))

        all_pairs = {(a, b) for a in comp_ids for b in comp_ids}
        neg_candidates = list(all_pairs - pos_pairs)

        target = len(dataset_pos) * 2
        random.shuffle(neg_candidates)
        neg_samples = neg_candidates[:target]

        dataset_neg = pd.DataFrame(neg_samples, columns=["comp1", "comp2"])
        dataset_neg["label"] = "no_relation"

        dataset = (
            pd.concat([dataset_pos, dataset_neg], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

        dataset['comp1'] = dataset['comp1'].apply(
            lambda x: df_components[df_components["id"] == x]['component'].values[0])
        dataset['comp2'] = dataset['comp2'].apply(
            lambda x: df_components[df_components["id"] == x]['component'].values[0])

        dataset['text'] = dataset['comp1'] + " [SEP] " + dataset['comp2']
        dataset['text'] = dataset['text'].apply(lambda x: x.strip().replace("\r", "").replace("\n", " "))
        dataset = dataset[['text', 'label']].dropna().drop_duplicates().reset_index(drop=True)

        labels2id = {key: idx for idx, key in enumerate(sorted(set(df_components["label"])))}
        dataset['label_value'] = dataset['label'].copy()
        dataset['label'] = dataset['label'].map(labels2id)

        train, test = train_test_split(dataset, test_size=0.2, random_state=42)
        dev, test = train_test_split(test, test_size=0.5, random_state=42)

        out_dir = folder % "multi_class_classification"
        train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        dev.to_csv(os.path.join(out_dir, "dev.csv"), index=False)
        test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    process_argument_detection(df.copy())
    relation_prediction(df.copy())
    os.remove(os.path.join(folder % "sequence_labelling", "base.csv"))


if __name__ == "__main__":
    root = rootutils.find_root("")
    process_aus_hansard(os.path.join(root, "data/binary_classification/AusHansard"))
    process_con_vote(os.path.join(root, "data/binary_classification/ConVote"))
    process_han_de_set(os.path.join(root, "data/binary_classification/HanDeSeT"))
    preprocess_parl_vote(os.path.join(root, "data/binary_classification/ParlVote"))
    preprocess_parl_vote_plus(os.path.join(root, "data/multi_class_classification/ParlVote+"))
    preprocess_elecdeb60to20_relations(os.path.join(root, "data/multi_class_classification/ElecDeb60to20-relations"))
    preprocess_motion_policy_preferences(os.path.join(root, "data/multi_class_classification/MotionPolicyPreference"))
    preprocess_elecdeb60to20_components(os.path.join(root, "data/sequence_labelling/ElecDeb60to20-components"))
    preprocess_ArgUNSC(os.path.join(root, "data/%s/ArgUNSC"))
