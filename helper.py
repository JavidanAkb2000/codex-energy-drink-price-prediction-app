from pathlib import Path
import json
import os
import pandas as pd
import numpy as np
import joblib
import sklearn
import lightgbm

# -----------------------
# Paths
# -----------------------
path_to_artifacts = Path(__file__).parent.resolve() / "model_artifacts"

# -----------------------
# Helper functions
# -----------------------
def read_encoding_json(filename):
    path = os.path.join(path_to_artifacts, filename)
    with open(path, "r") as f:
        return json.load(f)

# -----------------------
# Load model-related info
# -----------------------
encoding_info = read_encoding_json("encoding_info.json")
all_features_encoded = encoding_info['final_feature_list']

# Original columns (target column 'price_range' excluded)
original_columns = [
    'age', 'gender', 'zone', 'occupation',
    'income_levels', 'consume_frequency(weekly)', 'current_brand', 'preferable_consumption_size',
    'awareness_of_other_brands', 'reasons_for_choosing_brands', 'flavor_preference', 'purchase_channel',
    'packaging_preference', 'health_concerns', 'typical_consumption_situations'
]

# -----------------------
# Mapping of columns to input options (except age)
# -----------------------
input_options = {
    'gender': ['M', 'F'],
    'zone': ['Urban', 'Metro', 'Rural', 'Semi-Urban'],
    'occupation': ['Working Professional', 'Student', 'Entrepreneur', 'Retired'],
    'income_levels': ['<10L', '10L - 15L', '16L - 25L', '26L - 35L', '> 35L', 'Not Reported'],
    'consume_frequency(weekly)': ['0-2 times', '3-4 times', '5-7 times'],
    'current_brand': ['Newcomer', 'Established'],
    'preferable_consumption_size': ['Small (250 ml)', 'Medium (500 ml)', 'Large (1 L)'],
    'awareness_of_other_brands': ['0 to 1', '2 to 4', 'above 4'],
    'reasons_for_choosing_brands': ['Price', 'Quality', 'Availability', 'Brand Reputation'],
    'flavor_preference': ['Traditional', 'Exotic'],
    'purchase_channel': ['Online', 'Retail Store'],
    'packaging_preference': ['Simple', 'Premium', 'Eco-Friendly'],
    'health_concerns': [
        'Low (Not very concerned)',
        'Medium (Moderately health-conscious)',
        'High (Very health-conscious)'
    ],
    'typical_consumption_situations': [
        'Active (eg. Sports, gym)',
        'Social (eg. Parties)',
        'Casual (eg. At home)'
    ]
}

# -----------------------
# Load artifacts once
# -----------------------
ordinal_encoder = joblib.load(path_to_artifacts / "ordinal_encoder.pkl")
label_encoder_target = joblib.load(path_to_artifacts / "label_encoder_target.pkl")
lgbm_model = joblib.load(path_to_artifacts / "lightgbm_model.pkl")


# -----------------------
# Functions
# -----------------------

def make_dataframe(user_inp: dict, original_cols: list = original_columns):
    """Create a single-row DataFrame from user input."""
    row = {col: user_inp.get(col, None) for col in original_cols}
    df = pd.DataFrame([row], columns=original_cols)
    return df


def prepare_dataframe(df):
    """Feature engineering and column transformations."""
    df['age_group'] = pd.cut(
        df['age'],
        bins=[18, 25, 35, 45, 55, 70],
        labels=['18-25', '26-35', '36-45', '46-55', '56-70'],
        right=True,
        include_lowest=True
    )
    df = df.drop(columns=['age'])

    df['consume_frequency_weekly_encoded'] = df['consume_frequency(weekly)'].map({
        '0-2 times': 1,
        '3-4 times': 2,
        '5-7 times': 3
    })

    df['awareness_of_other_brands_encoded'] = df['awareness_of_other_brands'].map({
        '0 to 1': 1,
        '2 to 4': 2,
        'above 4': 3
    })

    df['cf_ab_score'] = (
        df['consume_frequency_weekly_encoded'] /
        (df['awareness_of_other_brands_encoded'] + df['consume_frequency_weekly_encoded'])
    ).round(2)

    df = df.drop(columns=['consume_frequency(weekly)', 'awareness_of_other_brands'])

    df['zone_encoded'] = df['zone'].map({
        'Rural': 1,
        'Semi-Urban': 2,
        'Urban': 3,
        'Metro': 4
    })

    df['income_levels_encoded'] = df['income_levels'].map({
        '<10L': 1,
        '10L - 15L': 2,
        '16L - 25L': 3,
        '26L - 35L': 4,
        '> 35L': 5,
        'Not Reported': 0
    })

    df['zas_score'] = df['zone_encoded'] * df['income_levels_encoded']

    df['bsi'] = np.where(
        (df['current_brand'] != 'Established') &
        (df['reasons_for_choosing_brands'].isin(['Price', 'Quality'])),
        1,
        0
    )

    df = df.drop(columns=['zone', 'income_levels'])

    df["loyalty_score"] = (
        df["consume_frequency_weekly_encoded"] / (df["awareness_of_other_brands_encoded"] + 1)
    ).clip(upper=2).round(2)

    records_to_drop = df[(df.age_group == '56-70') & (df.occupation == 'Student')].index
    df = df.drop(records_to_drop, axis=0)

    return df


def encode_and_align(df):
    """Ordinal and one-hot encoding + align columns to training features."""
    ordinal_cols = ['age_group', 'health_concerns', 'preferable_consumption_size']
    df[ordinal_cols] = ordinal_encoder.transform(df[ordinal_cols]) + 1

    onehot_cols = encoding_info['cols_to_onehot']
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    final_cols = encoding_info['final_feature_list']
    df = df.reindex(columns=final_cols, fill_value=0)

    return df


def validate_user_input(user_input: dict):
    """Ensure no impossible combinations like old age + student occupation."""
    age = user_input.get('age')
    occupation = user_input.get('occupation')

    if age is not None and occupation is not None:
        if age > 55 and occupation == 'Student':
            # Autocorrect or raise a clear error
            raise ValueError(f"Occupation 'Student' not allowed for age {age}.")
            # OR autocorrect to 'Working Professional' or default:
            # user_input['occupation'] = 'Working Professional'

    return user_input


def predict(user_input):
    """Run the full prediction pipeline for one user input."""
    # Create a fresh DataFrame every time
    user_input = validate_user_input(user_input)
    df = make_dataframe(user_input)
    df = prepare_dataframe(df)
    df = encode_and_align(df)

    prediction_encoded = lgbm_model.predict(df)
    prediction_label = label_encoder_target.inverse_transform(prediction_encoded)

    return prediction_label[0]


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    user_input_example = {
        "age": 30,
        "gender": "M",
        "zone": "Metro",
        "occupation": "Entrepreneur",
        "income_levels": "16L - 25L",
        "consume_frequency(weekly)": "5-7 times",
        "current_brand": "Established",
        "preferable_consumption_size": "Large (1 L)",
        "awareness_of_other_brands": "2 to 4",
        "reasons_for_choosing_brands": "Quality",
        "flavor_preference": "Traditional",
        "purchase_channel": "Online",
        "packaging_preference": "Premium",
        "health_concerns": "Medium (Moderately health-conscious)",
        "typical_consumption_situations": "Active (eg. Sports, gym)"
    }

    result = predict(user_input_example)
    print("Predicted price range:", result)

#  'gender': ['M', 'F'],
#     'zone': ['Urban', 'Metro', 'Rural', 'Semi-Urban'],
#     'occupation': ['Working Professional', 'Student', 'Entrepreneur', 'Retired'],
#     'income_levels': ['<10L', '10L - 15L', '16L - 25L', '26L - 35L', '> 35L', 'Not Reported'],
#     'consume_frequency(weekly)': ['0-2 times', '3-4 times', '5-7 times'],
#     'current_brand': ['Newcomer', 'Established'],
#     'preferable_consumption_size': ['Small (250 ml)', 'Medium (500 ml)', 'Large (1 L)'],
#     'awareness_of_other_brands': ['0 to 1', '2 to 4', 'above 4'],
#     'reasons_for_choosing_brands': ['Price', 'Quality', 'Availability', 'Brand Reputation'],
#     'flavor_preference': ['Traditional', 'Exotic'],
#     'purchase_channel': ['Online', 'Retail Store'],
#     'packaging_preference': ['Simple', 'Premium', 'Eco-Friendly'],
#     'health_concerns': [
#         'Low (Not very concerned)',
#         'Medium (Moderately health-conscious)',
#         'High (Very health-conscious)'
#     ],
#     'typical_consumption_situations': [
#         'Active (eg. Sports, gym)',
#         'Social (eg. Parties)',
#         'Casual (eg. At home)'
#     ]
# }
