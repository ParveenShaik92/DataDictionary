import pandas as pd
from datetime import datetime
from dateutil.parser import parse

def is_int(val):
    try:
        if pd.isna(val) or val == '':
            return True
        return float(val).is_integer()
    except:
        return False

def is_float(val):
    try:
        if pd.isna(val) or val == '':
            return True
        float(val)
        return True
    except:
        return False

def is_bool(val):
    return str(val).strip().lower() in ['true', 'false', '', 'nan']

def is_date(val):
    try:
        parse(val)
        return True
    except(ValueError, TypeError):
        return False

# Type inference logic
def infer_column_type(series, threshold=0):
    values = series.dropna().astype(str).tolist()
    total = len(values)
    if total == 0:
        return 'str'

    type_counts = {
        'int': 0,
        'float': 0,
        'bool': 0,
        'datetime': 0,
        'str': 0
    }

    for v in values:
        if is_bool(v):
            type_counts['bool'] += 1
        elif is_int(v):
            type_counts['int'] += 1
        elif is_float(v):
            type_counts['float'] += 1
        elif is_date(v):
            type_counts['datetime'] += 1
        else:
            type_counts['str'] += 1

        # print(v)
        # print(type_counts)

    # Get the type with the highest count
    dominant_type, count = max(type_counts.items(), key=lambda x: x[1])

    # Only return the dominant type if it crosses the threshold
    if count / total >= threshold:
        return dominant_type
    else:
        return 'str'


# Apply to all columns
#inferred_types = {col: infer_column_type(df[col]) for col in df.columns}

# Show result
#for col, inferred in inferred_types.items():
    #print(f"{col}: {inferred}")
