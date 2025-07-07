import spacy
import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
from pprint import pprint

import utils.helpers as dd_helpers
import utils.custom_ner_components as dd_ner_components

from transformers import pipeline

def data_processor(data: list, columns: list) -> list:
    """
    Function to process data.
    In a real scenario, this would contain logic to process the data.
    """
    processed_data = []

    # Load spaCy English model
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_core_web_trf")
    nlp = dd_ner_components.setup_gender_ner_component(nlp)


    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)
    df = df.replace('', np.nan)
    # df = df.replace('NULL', np.nan)
    missing_counts = df.isnull().sum()
    # add missing counts to processed data
    processed_data.append({"missing_counts": missing_counts.to_dict()})

    # Use threshold-based categorical detection
    threshold = dynamic_threshold_percentile(df)
    processed_data.append({"Detected_threshold": threshold})
    categorical_cols = detect_categorical_by_uniqueness(df, threshold)
    processed_data.append({"categorical_columns": categorical_cols})
    
    inferred_types = {col: dd_helpers.infer_column_type(df[col]) for col in df.columns}
    # Add inferred types to processed data
    processed_data.append({"inferred_types": inferred_types})
    for col in df.columns:
        if inferred_types[col] == 'int' or inferred_types[col] == 'float':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    summary = df.describe(include='all')
    summary_clean = summary.fillna("").astype(str).to_dict()
    processed_data.append({"data_description": summary_clean})
    
    col_ents = defaultdict(list)
    row_ents = []
    row_ents_text = set()
    print('Detecting Row and Column level metadata')
    for row in data:
        # Row NLP
        rowData = " ".join(f"{col} - {val}" for col, val in zip(columns, row))
        row_ents_result = nlp(str(rowData))
        for entItem in row_ents_result.ents:
            if entItem.text not in row_ents_text:
                row_ents.append(entItem.label_)
                row_ents_text.add(entItem.text)
        
        # Column NLP
        for column, dataItem in zip(columns, row):
            col_ents_result = nlp(dataItem)
            # if (column == 'column5' ) :
            #    print(f"===== {dataItem} ====")
            if not col_ents_result.ents:
                col_ents[column].append('UNKNOWN')
                # if (column == 'last_name' ) :
                #     print(f"MissingText: {dataItem}")
            else:
                for entity in col_ents_result.ents:
                    # if (column == 'last_name' ) :
                    #     print(f"Text: {entity.text}, Label: {entity.label_}")
                    if (entity.label_ == 'DATE') :
                        if (dd_helpers.is_date(entity.text) ):
                            col_ents[column].append(entity.label_)
                    elif (entity.label_ == 'PERSON'):
                        if not str(entity.text).isdigit():
                            col_ents[column].append(entity.label_)
                    else:
                        col_ents[column].append(entity.label_)
            # pprint(col_ents)
        print("-", end='', flush=True)

    # add detected entity types to processed data with column names
    processed_data.append({"detected_ner_column": {key: Counter(value).most_common(1) for key, value in col_ents.items()}})

    print('\n')
    print('Detecting provided dataset metadata')
    row_ents_count = Counter(row_ents)
    processed_data.append({"detected_ner_rows": row_ents_count })
    #pprint(row_ents_count);
    processed_data.append({"detected_dataset": infer_csv_topic_zero_shot_batch(df, columns, row_ents_count)})


#     for index, (key, value)  in enumerate(col_ents.items()):
#         lable_count = Counter(value)
#         print(f"Detected Entity Type for {key}")
#         print(lable_count.most_common(1))
    # Return processed data
    return processed_data


def infer_csv_topic_zero_shot_batch(df: pd.DataFrame, columns: list, ents: Counter[str]) -> list:
    candidate_labels = []
    ner_to_category = {
        "PERSON": "person information",
        "ORG": "employee records",
        "GPE": "geographic information",
        "LOC": "geographic information",
        "NORP": "demographic data",
        "FAC": "real estate listings",
        "PRODUCT": "product catalog",
        "EVENT": "survey responses",
        "WORK_OF_ART": "research publications",
        "LAW": "legal case records",
        "LANGUAGE": "academic records",
        "DATE": "financial data",
        "TIME": "sensor data",
        "PERCENT": "marketing data",
        "MONEY": "financial data",
        "QUANTITY": "inventory data",
        "ORDINAL": "academic records",
        "CARDINAL": "order details"
    }
    for ne_item, count in ents.items():
        candidate_labels.append(ner_to_category.get(ne_item, 'Unknown'))

    #pprint(candidate_labels)
    # Load zero-shot classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Store predictions
    predictions = []

    batch_size=10

    # Process in batches of batch_size
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start+batch_size].astype(str).values.flatten().tolist()
        text = "Headers: " + ", ".join(columns) + ". Sample values: " + ", ".join(batch)

        try:
            result = classifier(text, candidate_labels)
            predictions.append(result['labels'][0])  # take top predicted label
            # print(text)
            # print(result['labels'][0])
            print("-", end='', flush=True)
        except Exception as e:
            print(f"Error processing batch {start}-{start+batch_size}: {e}")

    # Count the most common predicted category
    most_common_label = Counter(predictions).most_common(1)[0][0] if predictions else "Unknown"

    return {
        "common_label" : most_common_label, 
        "Count": Counter(predictions)
        }
def detect_categorical_by_uniqueness(df: pd.DataFrame, threshold: float = 0.05) -> list:
    """
    Detects categorical columns using a uniqueness ratio heuristic:
    Columns with (nunique / total rows) < threshold are treated as categorical.
    
    :param df: Input DataFrame
    :param threshold: Ratio threshold (e.g. 0.05 for 5%)
    :return: List of likely categorical column names
    """
    if df.empty:
        return []

    categorical_cols = []

    for col in df.columns:
        ratio = df[col].nunique(dropna=True) / len(df)
        # print(f"Column: {col}")
        # print(f"Unique values: {df[col].nunique(dropna=True)}")
        # print(f"Total rows: {len(df)}")
        # print(f"Ratio: {ratio:.4f}")
        # print(f"Threshold: {threshold}")
        # print("*****")
        
        if ratio < threshold:
            categorical_cols.append(col)

    return categorical_cols

def dynamic_threshold_percentile(df: pd.DataFrame) -> float:
    ratios = {
    col: df[col].nunique(dropna=True) / len(df)
    for col in df.columns
    }
    ratios_array = np.array(list(ratios.values()))
    threshold = np.percentile(ratios_array, 25)
    return threshold