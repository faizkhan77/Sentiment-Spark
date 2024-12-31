import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import numpy as np
import torch
import time
from PIL import Image
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Paths
model_path = "artifacts/bert-model"
tokenizer_path = "artifacts/bert-tokenizer"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None


def analyze_text(text, threshold=3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    label = torch.argmax(outputs.logits) + 1
    classification = "Positive" if label >= threshold else "Negative"
    return label, classification, probabilities


# Load animations
lottie_analysis = load_lottie_url(
    "https://assets4.lottiefiles.com/packages/lf20_t24tpvcu.json"
)
lottie_visualization = load_lottie_url(
    "https://assets9.lottiefiles.com/packages/lf20_z9ed2jna.json"
)


# App title
st.set_page_config(page_title="Emotion Sense", page_icon="🙌", layout="wide")

st.title("Sentiment Spark: AI-Powered Sentiment Explorer 🚀")
st.markdown(
    """
    ### Dive into the power of AI to understand emotions! 
    Effortlessly analyze sentiments from text data and uncover trends through interactive visualizations.
    
    *Crafted for innovation and insight.*
    """
)


# hide streamlit's default style
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# Dark-Themed Sidebar menu
with st.sidebar:
    st.markdown(
        """
        <div style="background-color:#1e1e2f; padding:15px; border-radius:10px 10px 0 0;">
            <h2 style="color:#f0f0f0; text-align:center;">Sentiment Spark</h2>
            <p style="color:#a3a3a3; text-align:center; font-size:14px;">
                Analyze texts or datasets and visualize sentiment insights with ease.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        "Navigation",
        ["Analyze Text", "Visualize Insights"],
        icons=["chat-left-text", "bar-chart-line"],
        menu_icon="menu-button-fill",
        default_index=0,
        styles={
            "container": {
                "padding": "15px",
                "background-color": "#1e1e2f",
                "border-radius": "0 0 10px 10px;",
            },
            "icon": {"color": "#f05454", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "10px",
                "padding": "10px",
                "border-radius": "5px",
                "color": "#d4d4d4",
            },
            "nav-link-selected": {
                "background-color": "#f05454",
                "color": "white",
                "font-weight": "bold",
            },
        },
    )
    st.markdown(
        """
        <hr style="border: 1px solid #333; margin: 20px 0;">
        <p style="text-align:center; font-size:12px; color:#7a7a7a;">
            © 2024 Sentiment Spark | Mohd Faiz Khan
        </p>
        """,
        unsafe_allow_html=True,
    )


# Tab 1: Text Analysis
if selected == "Analyze Text":
    st_lottie(lottie_analysis, height=200, key="analysis")
    st.header("Sentiment Analysis")
    st.write(
        "Upload your text data or enter a single sentence to analyze its sentiment using advanced AI techniques."
    )

    # Input options
    option = st.radio(
        "Choose your input type:",
        ("Single Sentence", "Upload DataFrame"),
        horizontal=True,
        label_visibility="visible",
    )

    if option == "Single Sentence":
        st.header("🌟 Analyze a Single Sentence")
        st.write(
            "Enter a sentence to analyze its sentiment. The sentiment will be categorized "
            "as **Positive** or **Negative** based on a threshold. Additionally, probabilities for all labels will be displayed."
        )
        sentence = st.text_area(
            "Enter your sentence here:",
            placeholder="e.g., I love using AI for analysis!",
        )
        threshold = st.slider(
            "Set Positive Sentiment Threshold", min_value=1, max_value=5, value=3
        )

        # Analyze Sentiment Button with Loading State
        analyze_button = st.button("Analyze Sentiment")
        if analyze_button:
            if sentence.strip():
                with st.spinner("Analyzing the sentiment..."):
                    # Simulate a delay (e.g., for model prediction)

                    time.sleep(2)  # Replace with actual model processing time
                    label, classification, probabilities = analyze_text(
                        sentence, threshold
                    )

                    # Display Result
                    st.success("Sentiment analysis complete! 🎉")
                    st.write(f"### Sentiment: **{classification}**")
                    st.write(
                        f"### Star Rating: {'⭐' * label} ({label} Star{'s' if label > 1 else ''})"
                    )

                    # Display Probabilities
                    st.write("### Probability Distribution")
                    sorted_probs = sorted(
                        [(i + 1, p) for i, p in enumerate(probabilities)],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    for label, prob in sorted_probs:
                        st.write(f"**{label} Star**: {prob:.2%}")
            else:
                st.error("Please enter a sentence to analyze.")

    elif option == "Upload DataFrame":
        st.header("📊 Analyze Multiple Sentences")
        st.write(
            "Upload a file containing sentences (CSV or Excel) to analyze the sentiment of multiple texts at once. "
            "Ensure the column name for the text data contains the word 'text'."
        )
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])
        threshold = st.slider(
            "Set Positive Sentiment Threshold", min_value=1, max_value=5, value=3
        )

        if uploaded_file:
            try:
                with st.spinner(
                    "Please wait, analyzing and predicting the data. The time depends on the size of your data and internet speed..."
                ):
                    # Load data
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    # Match columns containing 'text' in their name (case-insensitive)
                    text_column = None
                    for col in df.columns:
                        if pd.Series(col).str.contains("text", case=False).any():
                            text_column = col
                            break

                    if text_column is None:
                        st.error(
                            "The uploaded file must contain a column with the word 'text' in the name."
                        )
                    else:
                        # Analyze each text
                        df["sentiment"], df["classification"], df["probabilities"] = (
                            zip(*df["text"].map(lambda x: analyze_text(x, threshold)))
                        )
                        # Create 5 columns for the probabilities of each sentiment class (1 to 5) as percentage values
                        for i in range(5):
                            # Store probabilities as numeric values (floats)
                            df[f"{i+1}"] = df["probabilities"].apply(
                                lambda x: x[i] * 100 if i < len(x) else 0
                            )

                        # Drop the old 'probabilities' column
                        df.drop(columns=["probabilities"], inplace=True)

                        # Save the analyzed DataFrame to session state
                        st.session_state.analyzed_df = df

                        # Display Results
                        st.success("Batch sentiment analysis complete! 🎉")
                        st.write("### Sentiment Analysis Results")
                        st.dataframe(df)
                        st.download_button(
                            label="Download Results as CSV",
                            data=df.to_csv(index=False),
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv",
                        )

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Tab 2: Visualization
if selected == "Visualize Insights":
    st_lottie(lottie_visualization, height=200, key="visualization")
    st.header("Sentiment Visualization Dashboard")
    st.write(
        "Explore interactive charts and graphs showcasing sentiment distribution and trends from your analyzed data."
    )

    # Show a loading spinner while the charts are being created
    with st.spinner("Creating Visualizations..."):
        time.sleep(2)  # Simulate loading time (remove in real scenario)

        # Check if analyzed data is available in session state
        if "analyzed_df" in st.session_state:
            df = st.session_state.analyzed_df

            # Ensure the sentiment column is numeric (extract value from tensor)
            df["sentiment"] = df["sentiment"].apply(
                lambda x: x.item() if isinstance(x, torch.Tensor) else x
            )

            # Make sure the sentiment values are valid (remove rows with invalid sentiment)
            df = df[df["sentiment"].notnull()]

            # --- Sentiment Distribution (Pie Chart) ---
            sentiment_counts = df["classification"].value_counts()
            sentiment_pie = px.pie(
                names=sentiment_counts.index,
                values=sentiment_counts.values,
                title="Sentiment Distribution",
                labels={"value": "Count", "classification": "Sentiment"},
            )

            # --- Sentiment Breakdown (Bar Chart) ---
            sentiment_labels = df["sentiment"].value_counts()
            sentiment_bar = px.bar(
                sentiment_labels,
                x=sentiment_labels.index,
                y=sentiment_labels.values,
                title="Sentiment Breakdown by Classification",
                labels={"sentiment": "Sentiment (Stars)", "value": "Count"},
                category_orders={
                    "sentiment": ["1", "2", "3", "4", "5"]
                },  # Sorting by star rating
            )

            # --- Probability Distribution per Sentiment (Stacked Bar Chart) ---
            prob_columns = ["1", "2", "3", "4", "5"]
            prob_df = df[prob_columns].mean().reset_index()
            prob_df.columns = ["Star Rating", "Probability (%)"]
            prob_df["Probability (%)"] *= 100  # Convert to percentage

            prob_stacked_bar = px.bar(
                prob_df,
                x="Star Rating",
                y="Probability (%)",
                title="Average Probability Distribution per Sentiment",
                labels={
                    "Star Rating": "Sentiment (Stars)",
                    "Probability (%)": "Average Probability (%)",
                },
                color="Star Rating",
                color_discrete_map={
                    str(
                        i
                    ): f"rgb({max(0, min(255, i*50))}, {max(0, min(255, 100-i*30))}, {max(0, min(255, i*20))})"
                    for i in range(1, 6)
                },
            )

            # Displaying the Charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sentiment Distribution")
                st.plotly_chart(sentiment_pie, use_container_width=True)

            with col2:
                st.subheader("Sentiment Breakdown by Classification")
                st.plotly_chart(sentiment_bar, use_container_width=True)

            st.subheader("Average Probability Distribution per Sentiment")
            st.plotly_chart(prob_stacked_bar, use_container_width=True)

        else:
            st.error("No analyzed data found. Please perform batch analysis first.")

# Footer
st.markdown(
    "---\n**Emotion Sense** | Built with ❤️ to empower your sentiment analysis journey."
)
