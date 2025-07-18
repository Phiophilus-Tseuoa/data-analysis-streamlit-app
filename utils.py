
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def clean_column_names(df, new_names):
    return df.rename(columns=new_names)

def replace_missing_values(df):
    df.replace("?", np.nan, inplace=True)
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].astype(float)
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def drop_columns(df, columns_to_drop):
    df.drop(columns=columns_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def convert_column_types(df, type_map):
    for col, col_type in type_map.items():
        if col_type != 'No Change':
            try:
                df[col] = df[col].astype(col_type)
            except Exception as e:
                print(f"Could not convert {col} to {col_type}: {e}")
    return df

def scale_data(df, columns, method):
    try:
        if method == "Standardization (Z-score)":
            df[columns] = StandardScaler().fit_transform(df[columns])
        elif method == "Min-Max Scaling":
            df[columns] = MinMaxScaler().fit_transform(df[columns])
        elif method == "Simple Feature Scaling":
            df[columns] = df[columns] / df[columns].max()
    except Exception as e:
        print(f"Scaling error: {e}")
    return df

def apply_binning(df, column, bins):
    df[f"{column}_binned"] = pd.cut(df[column], bins=bins)
    return df

def train_model(X, y, model_type="Logistic Regression", max_depth=3):
    if model_type == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = DecisionTreeClassifier(max_depth=max_depth)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    return model, X_test, y_test, y_pred, metrics

def encode_target(y):
    if y.dtype == 'object':
        return LabelEncoder().fit_transform(y)
    return y
