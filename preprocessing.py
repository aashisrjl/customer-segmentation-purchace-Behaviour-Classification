import warnings

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')


def preprocess_data(
    input_path: str = 'data/raw/raw_data.csv',
    output_path: str = 'data/processed/processed_data.csv',
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    df = df.drop(columns=['ID'], errors='ignore')

    num_cols = df.select_dtypes(include='number').columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    outlier_cols = ['Age', 'Work_Experience', 'Family_Size']
    for col in outlier_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower=lower, upper=upper)

    le = LabelEncoder()

    if 'Spending_Score' in df.columns:
        spending_map = {'Low': 0, 'Average': 1, 'High': 2}
        df['Spending_Score'] = df['Spending_Score'].map(spending_map)

    binary_map = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0}
    for col in ['Gender', 'Ever_Married', 'Graduated']:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    df_scaled.to_csv(output_path, index=False)
    return df_scaled


if __name__ == '__main__':
    processed = preprocess_data()
    print(f'Saved processed_data.csv — shape: {processed.shape}')
