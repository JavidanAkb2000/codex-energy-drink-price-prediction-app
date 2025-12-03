import streamlit as st
import base64
from helper import input_options, predict

# Background image via CSS
def add_bg_from_local():
    # Read and encode the local image
    with open("model_artifacts/img/codex_back.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
        }}

        /* Title styling */
        h1 {{
            color: white !important;
            text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.9);
            text-align: center;
            font-size: 3rem !important;
            margin-bottom: 0.5rem !important;
        }}

        /* Subtitle styling */
        .stApp > header + div > div > div > div > div:nth-child(2) {{
            text-align: center;
        }}

        h2, h3 {{
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
        }}

        /* Container for form sections */
        .block-container {{
            background-color: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem !important;
            margin-top: 2rem;
        }}

        /* Input field containers */
        div[data-testid="stVerticalBlock"] > div:has(div.stSelectbox),
        div[data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {{
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}

        /* Label styling */
        label, .css-1y4p8pa {{
            color: white !important;
            font-weight: 600 !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        }}

        /* Input fields */
        input, select {{
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: black !important;
            border-radius: 5px !important;
        }}

        /* Button styling */
        .stButton > button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            font-size: 1.2rem;
            padding: 0.75rem 2rem;
            border-radius: 10px;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 1rem;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        }}

        /* Success message */
        .element-container:has(.stSuccess) {{
            background-color: rgba(40, 167, 69, 0.2);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 1rem;
            border: 2px solid rgba(40, 167, 69, 0.5);
        }}

        /* Section headers */
        .stMarkdown h3 {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.6) 0%, rgba(118, 75, 162, 0.6) 100%);
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_bg_from_local()

# Title with icon
st.title("🧃 CodeX Energy Drink: Price Prediction")
st.subheader("Predict the optimal price for your energy drink based on individual information's")


# User inputs
st.markdown("### 👤 Individual Information's")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, step=1, value=25)
    gender = st.selectbox("Gender", input_options['gender'])
    zone = st.selectbox("Zone", input_options['zone'])

with col2:
    occupation = st.selectbox("Occupation", input_options['occupation'])
    income = st.selectbox("Income Level (INR)", input_options['income_levels'])

st.markdown("### 🧃 Consumption Preferences")
col3, col4 = st.columns(2)

with col3:
    freq = st.selectbox("Consumption Frequency (weekly)", input_options['consume_frequency(weekly)'])
    brand = st.selectbox("Current Brand", input_options['current_brand'])
    size = st.selectbox("Preferred Size", input_options['preferable_consumption_size'])
    awareness = st.selectbox("Awareness of Other Brands", input_options['awareness_of_other_brands'])
    reason = st.selectbox("Reason for Choosing Brand", input_options['reasons_for_choosing_brands'])

with col4:
    flavor = st.selectbox("Flavor Preference", input_options['flavor_preference'])
    channel = st.selectbox("Purchase Channel", input_options['purchase_channel'])
    packaging = st.selectbox("Packaging Preference", input_options['packaging_preference'])
    health = st.selectbox("Health Concerns", input_options['health_concerns'])
    situation = st.selectbox("Typical Consumption Situation", input_options['typical_consumption_situations'])

# Predict button
if st.button("⚡ Calculate Price Range"):
    # Collect all user inputs in a dictionary
    user_input = {
        'age': age,
        'gender': gender,
        'zone': zone,
        'occupation': occupation,
        'income_levels': income,
        'consume_frequency(weekly)': freq,
        'current_brand': brand,
        'preferable_consumption_size': size,
        'awareness_of_other_brands': awareness,
        'reasons_for_choosing_brands': reason,
        'flavor_preference': flavor,
        'purchase_channel': channel,
        'packaging_preference': packaging,
        'health_concerns': health,
        'typical_consumption_situations': situation
    }




    # Print all inputs
    # print("=" * 50)
    # print("USER INPUT DATA")
    # print("=" * 50)
    # for key, value in user_input.items():
    #     print(f"{key}: {value}")
    # print("=" * 50)



    # You can now send this to helper.py
    # from helper import predict_price
    # predicted_price = predict_price(user_input)

    # Placeholder for model prediction
    # predicted_price = "₹85 - ₹110"  # Dummy output
    try:
        predicted_price = predict(user_input)
        st.success(f"🎯 Predicted Price Range: **{predicted_price}** INR")
    except ValueError as e:
        st.error(f"❌ Input error: {e}")
