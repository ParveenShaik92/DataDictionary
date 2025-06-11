import spacy
import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
from pprint import pprint
from DataDictionary.utils.helpers import dd_helpers
from DataDictionary.utils.custom_ner_components import dd_ner_components

def data_processor(data: list) -> list:
    """
    Function to process data.
    In a real scenario, this would contain logic to process the data.
    """
    print(f"Mock processing data: {data[:2]}...")  # Print first 2 items for brevity
    processed_data = []

    # Load spaCy English model
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_core_web_trf")

    # Convert to DataFrame
    columns = ['column1', 'column2', 'column3']  # Example columns, adjust as needed
    df = pd.DataFrame(data, columns=columns)
    df = df.replace('', np.nan)
    # df = df.replace('NULL', np.nan)
    missing_counts = df.isnull().sum()
    # add missing counts to processed data
    processed_data.append({"missing_counts": missing_counts.to_dict()})

    inferred_types = {col: dd_helpers.infer_column_type(df[col]) for col in df.columns}
    # Add inferred types to processed data
    processed_data.append({"inferred_types": inferred_types})

    col1_ents = defaultdict(list)
    for row in rows:
        for column, dataItem in zip(columns, row):
            ent = nlp(str(dataItem))

            # if (column == 'column5' ) :
            #     print(f"===== {dataItem} ====")
                #pprint(ent.ents)
            for entity in ent.ents:
                # if (column == 'column5' ) :
                #     print(f"Text: {entity.text}, Label: {entity.label_}")
                if (entity.label_ == 'DATE') :
                    if (dd_helpers.is_date(entity.text) ):
                        col1_ents[column].append(entity.label_)
                elif (entity.label_ == 'PERSON'):
                    if not str(entity.text).isdigit():
                        col1_ents[column].append(entity.label_)
                else:
                    col1_ents[column].append(entity.label_)
            # pprint(col1_ents)

    # add detected entity types to processed data with column names
    processed_data.append({"detected_entities": {key: Counter(value).most_common(1) for key, value in col1_ents.items()}})

#     for index, (key, value)  in enumerate(col1_ents.items()):
#         lable_count = Counter(value)
#         print(f"Detected Entity Type for {key}")
#         print(lable_count.most_common(1))
    # Return processed data
    return processed_data