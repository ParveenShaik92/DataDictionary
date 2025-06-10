import sqlite3
import sqlite_utils
import spacy
import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
from pprint import pprint
import finddatatype 

# used to create new entity spans in the document.
from spacy.tokens import Span
# matches exact phrases efficiently, better than regex for known terms
from spacy.matcher import PhraseMatcher
from spacy.language import Language


# Load spaCy English model
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_trf")

###############################################
# Define/Add Named Entity label (like GENDER) in spaCy
###############################################

# Define terms to match as GENDER
gender_terms = ["Male", "Female", "M", "F"]
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(term) for term in gender_terms]
matcher.add("GENDER", patterns)

# Register the custom component with the @Language.component decorator
@Language.component("gender_entity_component")
def gender_entity_component(doc):
    matches = matcher(doc)
    new_ents = list(doc.ents)  # Keep existing entities

    for match_id, start, end in matches:
        span = Span(doc, start, end, label=nlp.vocab.strings["GENDER"])

        # Check if the entity is already part of the entities in doc.ents
        if not any([span.start >= ent.start and span.end <= ent.end for ent in doc.ents]):
            new_ents.append(span)

    # Ensure we only add valid, non-overlapping entities
    doc.ents = new_ents
    return doc

# Add custom component to the pipeline using its string name
nlp.add_pipe("gender_entity_component", after="ner")

###############################################
# END
###############################################

# Connect to (or create) a SQLite database
conn = sqlite3.connect('people.db')
cursor = conn.cursor()

# Create the table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS people3 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        column1 TEXT NOT NULL,
        column2 TEXT,
        column3 TEXT,
        column4 TEXT,
        column5 TEXT        
    )
''')

# # Insert a record
# Sample data with both valid and weird values
# people_data = [
#     ('Alice Smith', 'Female', 'USA','22-04-2010','Hindi'),
#     ('Bob Johnson', 'M', '??','10/09/1999','Telugu'),
#     ('654556', 'Male', 'UK','today','spanish'),
#     ('Diana Prince', 'F', 'France','09/29/22','??'),
#     ('Ethan Hunt', 'Male', 'OJIOHBJ','22/20','place'),
#     ('olkpoekf;ok', 'Female', 'Ireland','yesterday','Language'),
#     ('George Lucas', 'NA', 'USA','12/12/2024','Greek'),
#     ('Hannah Baker', 'F', 'JNJKJKN','10-10-2010','English'),
#     ('Hindi', 'Male', 'Russia','20 May 2015','Python'),
#     ('Julia Roberts', 'Female', 'USA','NULL','Tamil'),
#     ('sam','M','Japan','12-12-2020','French'),
#     ('Kiran@2','Female','US','30/12/2019','English')
# ]

# Insert multiple rows
# cursor.executemany('''
#     INSERT INTO people3 (column1, column2, column3, column4, column5)
#     VALUES (?, ?, ?, ?, ?)
# ''', people_data)

# #conn.commit()
# print("10 rows inserted successfully!")


cursor.execute('SELECT column1, column2, column3, column4, column5 FROM people3')
rows = cursor.fetchall()

columns = [desc[0] for desc in cursor.description]

# Convert to DataFrame
df = pd.DataFrame(rows, columns=columns)
df = df.replace('', np.nan)
# df = df.replace('NULL', np.nan)
missing_counts = df.isnull().sum()

print("Missing values per column:")
print(missing_counts)

#Apply to all columns
inferred_types = {col: finddatatype.infer_column_type(df[col]) for col in df.columns}

# Show result
for col, inferred in inferred_types.items():
    print(f"{col}: {inferred}")




#pprint(rows);


# Commit and close
conn.commit()
conn.close()


columns = ['column1', 'column2', 'column3', 'column4', 'column5']
col1_ents = defaultdict(list)
for row in rows:
    for column, data in zip(columns, row):
        
        ent = nlp(str(data))

        # if (column == 'column5' ) :
        #     print(f"===== {data} ====")
            #pprint(ent.ents)
        for entity in ent.ents:
            # if (column == 'column5' ) :
            #     print(f"Text: {entity.text}, Label: {entity.label_}")
            if (entity.label_ == 'DATE') :
                if (finddatatype.is_date(entity.text) ):
                    col1_ents[column].append(entity.label_)
            elif (entity.label_ == 'PERSON'):
                if not str(entity.text).isdigit():
                    col1_ents[column].append(entity.label_)
            else:
                col1_ents[column].append(entity.label_)
        # pprint(col1_ents)
for index, (key, value)  in enumerate(col1_ents.items()):
    lable_count = Counter(value)
    print(f"Detected Entity Type for {key}")
    print(lable_count.most_common(1))
    
    




