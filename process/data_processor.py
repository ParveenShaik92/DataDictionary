import spacy
import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
import math

import utils.helpers as dd_helpers
import utils.custom_ner_components as dd_ner_components
import utils.dd_genai as dd_genai

from transformers import pipeline

# pip install scikit-learn
from sklearn.model_selection import StratifiedShuffleSplit


def data_processor(data: list, columns: list) -> list:
    """
    Function to process data.
    In a real scenario, this would contain logic to process the data.
    """
    output = []

    # Load spaCy English model
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_core_web_trf")
    nlp = dd_ner_components.setup_gender_ner_component(nlp)

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)
    df = df.replace('', np.nan)
    
    # Finding missing count.
    missing_counts = df.isnull().sum()
    max_missing_count = max(missing_counts.to_dict().values())
    output.append({"Missing_Column_values_Count": missing_counts})

    # Use dynamic threshold-based categorical detection
    threshold = dynamic_threshold_percentile(df)
    output.append({"Detected_Threshold": threshold})
    categorical_cols = detect_categorical_by_dynamic_threshold(df, threshold)
    output.append({"Categorical_Columns": categorical_cols})

    # Get stratified_sample group by all categorial cols.
    StratifiedSamples = get_stratified_sample_using_categorical_cols(df, categorical_cols, max_missing_count)
    #output.append({"Stratified_Samples": StratifiedSamples})
    
    # Fill the missing values in categorial cols
    for col in df.columns:
        if col in StratifiedSamples.columns:
            df = fill_missing_categorical_with_sample(df, col, StratifiedSamples)
    df.to_csv("output_processed.csv", index=False, na_rep="NULL", encoding="utf-8")

    # Get AI sample from the processed dataset.
    AI_sample_count = len(df) * (5/100)
    AI_samples = get_stratified_sample_using_categorical_cols(df, categorical_cols, AI_sample_count)
    print(AI_samples)
    output.append({"AI_Samples": AI_samples})

    # AI-based summaries for the dataset
    # Prepare a Prompt for the Entire Dataset
    AISummary = {}
    dataset_summary_prompt = (
            "Analyze the following dataset structure and generate an overall summary or description:\n\n"
                f"{AI_samples.to_string(index=False)}\n\n"
                "Please describe the overall theme or content of this dataset."
            )
    #print(dataset_summary_prompt)
    response = dd_genai.get_gemini_response(dataset_summary_prompt)
    AISummary["dataset"] = response
    output.append({"AI_Dataset_Summary": AISummary})

    # Generate Summaries for Individual Columns
    AISummary = {}
    for column in AI_samples.columns:
        column_sample = AI_samples[column].dropna().astype(str).unique()[:10]
        values_text = ", ".join(column_sample)
        column_prompt = (
            f"Based on the sample values: {values_text}, "
            f"provide a brief description of what the column '{column}' likely represents in the dataset."
        )
        column_description = dd_genai.get_gemini_response(column_prompt)
        AISummary[column] = column_description
        #print(f"\nDescription for column '{column}':\n{column_description}")

    output.append({"AI_Columns_Summary": AISummary})

    # Finding inferred types
    inferred_types = {col: dd_helpers.infer_column_type(df[col]) for col in df.columns}
    output.append({"Inferred_Types": inferred_types})

    # Finding the column descriptions
    for col in df.columns:
        if inferred_types[col] == 'int' or inferred_types[col] == 'float':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    summary = df.describe(include='all')
    output.append({"Column_Descriptions": summary})

    # Detecting NER for cloumns and rows.
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

    detectedNER = {key: Counter(value).most_common(1) for key, value in col_ents.items()}
    output.append({"Column_NERs": detectedNER })
    

    # Detect Dataset descripton.
    print('\n')
    print('Detecting provided dataset metadata')
    row_ents_count = Counter(row_ents)
    output.append({"Rows_NERs": row_ents_count })

    output.append({"Detected_Dataset": infer_csv_topic_zero_shot_batch(df, columns, row_ents_count)})

    return output


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
def detect_categorical_by_dynamic_threshold(df: pd.DataFrame, threshold: float = 0.05) -> list:
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
        if ratio < threshold:
            categorical_cols.append(col)
    return categorical_cols

# Calculates a dynamic threshold using the 25th percentile of column uniqueness ratios to categorical columns.
def dynamic_threshold_percentile(df: pd.DataFrame) -> float:
    ratios = {
        col: df[col].nunique(dropna=True) / len(df)
        for col in df.columns
    }
    ratios_array = np.array(list(ratios.values()))
    threshold = np.percentile(ratios_array, 25)
    return threshold


def get_stratified_sample_using_categorical_cols(df, strata_cols, min_samples=10):
    """
    Returns a stratified sample using sklearn StratifiedShuffleSplit.
    Falls back to group-wise ceil-based sampling if output is empty or too small.

    Parameters:
        df (pd.DataFrame): Input dataset
        strata_cols (list): List of categorical columns used for stratification
        min_samples (int): Minimum number of total samples to return

    Returns:
        pd.DataFrame: Stratified sampled rows
    """
    try:
        if df.empty or not strata_cols:
            return pd.DataFrame()

        df = df.copy()
        df = df[df[strata_cols].apply(lambda row: all(pd.notna(row)) and all(str(x).strip() != '' for x in row), axis=1)]
        df['_strata'] = df[strata_cols].astype(str).agg('_'.join, axis=1)
        total_rows = len(df)

        # Step 1: Try StratifiedShuffleSplit
        fraction = min(1.0, max(0.01, min_samples / total_rows))
        #print(f"percentange : {fraction*100} %")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=42)

        try:
            for _, test_idx in sss.split(df, df['_strata']):
                stratified_sample = df.iloc[test_idx].drop(columns='_strata')
                if len(stratified_sample) >= min_samples:
                    return stratified_sample
        except Exception as e:
            print(f"Fallback - group-wise ceil-based sampling")

        # Step 2: Fallback - group-wise ceil-based sampling
        valid_strata = (
            df.groupby('_strata')
            .filter(lambda g: len(g) >= 2)  # OR your own condition
        )

        strata_counts = valid_strata['_strata'].value_counts()
        total_strata = len(strata_counts)

        if total_strata == 0:
            print("No valid strata found for sampling.")
            return pd.DataFrame()

        samples_per_stratum = math.ceil(min_samples / total_strata)
        result = (
            valid_strata.groupby('_strata', group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), samples_per_stratum), random_state=42))
            .drop(columns='_strata')
            .reset_index(drop=True)
        )

        return result

    except Exception as e:
        print({"Error": str(e)})
        return pd.DataFrame()

def fill_missing_categorical_with_sample(df: pd.DataFrame, column: str, stratified_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing or empty values in a categorical column using values
    from a stratified sample DataFrame.

    Parameters:
        df (pd.DataFrame): The original DataFrame.
        column (str): Categorical column to fill.
        stratified_sample (pd.DataFrame): DataFrame containing replacement values.

    Returns:
        pd.DataFrame: DataFrame with missing values in the column filled.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in DataFrame.")

    if column not in stratified_sample.columns:
        raise ValueError(f"Column '{column}' not found in stratified_sample.")

    # Filter out invalid replacement values from the stratified sample
    sample_values = stratified_sample[column].dropna()
    sample_values = sample_values[sample_values.astype(str).str.strip() != ""]

    if sample_values.empty:
        raise ValueError("No valid replacement values found in stratified_sample.")

    sample_values = sample_values.tolist()
    total_samples = len(sample_values)
    sample_index = 0

    df = df.copy()

    for idx, value in df[column].items():
        if pd.isna(value) or str(value).strip() == "":
            df.at[idx, column] = sample_values[sample_index % total_samples]
            sample_index += 1

    return df