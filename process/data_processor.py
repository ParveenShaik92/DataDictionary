import spacy
import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
from pprint import pprint

import utils.helpers as dd_helpers
import utils.custom_ner_components as dd_ner_components

from transformers import pipeline

# pip install scikit-learn
from sklearn.model_selection import train_test_split


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
    output.append({"missing_counts": missing_counts.to_dict()})

    # Use threshold-based categorical detection
    threshold = dynamic_threshold_percentile(df)
    output.append({"Detected_threshold": threshold})
    categorical_cols = detect_categorical_by_uniqueness(df, threshold)
    output.append({"categorical_columns": categorical_cols})

    # Find Stratified Sample for each categorial cols
    samples = stratified_samples_by_missing_count(df,categorical_cols)
    samples_serialized = {
        col: df_sample.to_dict(orient="records")
        for col, df_sample in samples.items()
    }
    output.append({"stratified_sample": samples_serialized})

    # Get stratified_sample group by all categorial cols.
    groupBySamples = get_stratified_sample_groupby(df,categorical_cols)
    output.append({"stratified_sample_groupby": groupBySamples})

    # Fill the missing values in categorial cols
    output_df = df.copy()
    for col, samples_data in samples_serialized.items():
        if samples_data:
            output_df = fill_missing_categorical_with_sample(output_df, col, samples_data)

    # Get Stratified Sample for each cols which are not categorial cols
    nonCategorialColumnSamples = {}
    for col in df.columns:
        samplesData = stratified_sample_by_column(df, col)
        samples_json_ready = samplesData.to_dict(orient="records")
        nonCategorialColumnSamples[col]= samples_json_ready
    output.append({"nonCategorialColumnSamples" : nonCategorialColumnSamples})

    # Fill the missing values using non categorical cols sample data.
    for col, samples_data in nonCategorialColumnSamples.items():
        if samples_data:
            output_df = fill_missing_categorical_with_sample(output_df, col, samples_data)

    output_df.to_csv("output_process.csv", index=False, na_rep="NULL", encoding="utf-8")

    # Finding inferred types
    inferred_types = {col: dd_helpers.infer_column_type(df[col]) for col in df.columns}
    output.append({"inferred_types": inferred_types})

    # Finding the column descriptions
    for col in df.columns:
        if inferred_types[col] == 'int' or inferred_types[col] == 'float':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    summary = df.describe(include='all')
    summary_clean = summary.fillna("").astype(str).to_dict()
    output.append({"data_description": summary_clean})

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

    output.append({"detected_ner_column": {key: Counter(value).most_common(1) for key, value in col_ents.items()}})

    # Detect Dataset descripton.
    print('\n')
    print('Detecting provided dataset metadata')
    row_ents_count = Counter(row_ents)
    output.append({"detected_ner_rows": row_ents_count })
    #pprint(row_ents_count);
    output.append({"detected_dataset": infer_csv_topic_zero_shot_batch(df, columns, row_ents_count)})


#     for index, (key, value)  in enumerate(col_ents.items()):
#         lable_count = Counter(value)
#         print(f"Detected Entity Type for {key}")
#         print(lable_count.most_common(1))
    # Return processed data
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

def stratified_samples_by_missing_count(df: pd.DataFrame, categorical_columns: list) -> dict:
    """
    Generate stratified samples based on the number of missing values in each categorical column.
    The sample size is equal to the number of missing values in that column.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - categorical_columns (list): List of categorical columns to stratify on.
    
    Returns:
    - dict: Dictionary with column names as keys and stratified sample DataFrames as values.
    """
    samples = {}

    for col in categorical_columns:
        # Replace missing with "MISSING" and convert all to string for consistent stratification
        stratify_col = df[col].fillna("MISSING").astype(str)
        
        # Count missing values
        missing_count = (df[col].isna()).sum()

        print(f"missing_count: {missing_count}")
        
        if missing_count == 0:
            continue  # Skip columns with no missing data

        # Use train_test_split to get a stratified sample
        try:
            sample_df, _ = train_test_split(
                df,
                train_size=missing_count,
                stratify=stratify_col,
                random_state=42
            )
            samples[col] = sample_df.reset_index(drop=True)
        except ValueError as e:
            print(f"Skipping column '{col}' due to stratification error: {e}")
            samples[col] = pd.DataFrame()

    return samples


def get_stratified_sample_groupby(df, strata_cols, fraction=0.1):
    """
    Returns a JSON-serializable stratified sample from the DataFrame.

    Parameters:
        df (pd.DataFrame): Input dataset
        strata_cols (list): List of categorical columns to stratify by
        fraction (float): Fraction to sample from each group (default: 0.1)

    Returns:
        list: Stratified sampled records as list of dictionaries (JSON-serializable)
    """
    try:
        stratified_sample = df.groupby(strata_cols, group_keys=False)\
                              .apply(lambda x: x.sample(frac=fraction, random_state=42, replace=True))\
                              .reset_index(drop=True)
        
        # Check if result is DataFrame before filling NaNs
        if isinstance(stratified_sample, pd.DataFrame):
            stratified_sample = stratified_sample.fillna(value=None)
            return stratified_sample.to_dict(orient='records')
        else:
            return []

    except Exception as e:
        print({"Message": str(e)})

def fill_missing_categorical_with_sample(df, column, stratified_sample):
    """
    Fills missing or empty values in a categorical column using values
    from a stratified sample.

    Parameters:
        df (pd.DataFrame): The original DataFrame
        column (str): Categorical column to fill
        stratified_sample (list of dict): List of dicts (from stratified sampling)

    Returns:
        pd.DataFrame: DataFrame with missing values filled
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in DataFrame.")

    # Extract replacement values from stratified sample
    sample_values = [row[column] for row in stratified_sample if row.get(column) not in [None, "", np.nan]]
    print(sample_values);

    if not sample_values:
        raise ValueError("No valid replacement values found in stratified_sample.")

    sample_index = 0
    total_samples = len(sample_values)

    # Copy the DataFrame to avoid changing the original
    df = df.copy()

    for idx, value in df[column].items():
        if pd.isna(value) or str(value).strip() == "":
            df.at[idx, column] = sample_values[sample_index % total_samples]
            sample_index += 1

    return df

def stratified_sample_by_column(df, column,random_state=None):
    n = (df[column].isna()).sum()
    return df.groupby(column, group_keys=False).apply(
        lambda x: x.sample(n=n, random_state=random_state) if len(x) >= n else x
    )