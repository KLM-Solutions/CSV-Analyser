import streamlit as st
import pandas as pd
import json
from openai import OpenAI
from datetime import datetime, timedelta
import os
import io
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
load_dotenv()

# Function to get OpenAI API key
def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            # Save the API key to .env file
            with open(".env", "a") as env_file:
                env_file.write(f"\nOPENAI_API_KEY={api_key}")
            st.success("API key saved to .env file.")
    return api_key

# Initialize OpenAI client
api_key = get_openai_api_key()
if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None

def csv_to_json(csv_file):
    try:
        df = pd.read_csv(csv_file)
        return df.to_json(orient="records")
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or contains no parsable data.")
        return None
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
        return None

def analyze_anomalies(df):
    anomalies = []

    # Implement anomaly detection logic based on the provided rules
    # Example: Forwarding Email to another account
    if 'RecordType' in df.columns and 'Operation' in df.columns and 'Parameter' in df.columns:
        forwarding_emails = df[(df['RecordType'] == 1) & 
                               (df['Operation'] == 'Set-Mailbox') & 
                               (df['Parameter'].str.contains('ForwardingSmtpAddress', na=False))]
        if not forwarding_emails.empty:
            anomalies.append("Forwarding Email to another account detected")

    # Implement other anomaly detection rules here...

    return anomalies

def chunk_dataframe(df, chunk_size=100):
    return [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

def get_most_relevant_chunks(query, chunks, top_n=5):
    # Convert chunks to strings
    chunk_texts = [chunk.to_string() for chunk in chunks]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform chunk texts
    chunk_vectors = vectorizer.fit_transform(chunk_texts)
    
    # Transform query
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_vector, chunk_vectors)
    
    # Get indices of top N similar chunks
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    
    return [chunks[i] for i in top_indices]

def summarize_dataframe(df):
    summary = []
    for column in df.columns:
        column_type = str(df[column].dtype)
        unique_values = df[column].nunique()
        summary.append(f"Column '{column}' (Type: {column_type}):")
        summary.append(f"  - Unique values: {unique_values}")
        if pd.api.types.is_numeric_dtype(df[column]):
            summary.append(f"  - Mean: {df[column].mean():.2f}")
            summary.append(f"  - Median: {df[column].median():.2f}")
            summary.append(f"  - Min: {df[column].min()}")
            summary.append(f"  - Max: {df[column].max()}")
        elif pd.api.types.is_string_dtype(df[column]):
            summary.append(f"  - Most common value: {df[column].mode().iloc[0]}")
    return "\n".join(summary)

def ask_openai(question, context, df_summary):
    if not client:
        st.error("OpenAI API key is not set. Please enter your API key.")
        return None
    
    try:
        max_context_length = 3000  # Adjust as needed
        truncated_context = context[:max_context_length]
        
        system_message = """
You are an AI assistant specialized in analyzing JSON data that was converted from a CSV file. Your task is to thoroughly analyze the content, generate summaries, and provide comprehensive reports based on user queries. give the response which is only related to the query.

When analyzing data or answering questions, folloe these below steps:

1. Clearly define the user's question and provide necessary context for the data you're analyzing.
   
2. Provide an overview of the data as relevant to the query. Summarize key points, patterns, statistics, and any significant trends, with explanations where necessary.

3. Dont just analyze individual values—analyze the content holistically. Interpret the relationships between data fields, check for inconsistencies or patterns, and note any significant findings.

4. Offer explanations for patterns, anomalies, or issues in the data. Suggest multiple interpretations or potential reasons, backed by data logic.

5. Suggest practical steps for addressing issues, optimizing processes, or further investigation areas where applicable. Provide suggestions on data improvement or further analysis.

6. Present responses in a clear, accessible tone. Use professional language but avoid unnecessary complexity unless required by the context.

Where helpful, use bullet points or numbered lists for clarity. If data insights are requested, present them in tables or statistical summaries. Aim for precision and clarity to ensure the report is easy to follow while being informative.

Your responses should be well-structured, and each report should be tailored to the specific query while leveraging the entire dataset for the analysis.
"""

        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Make sure to use an appropriate model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"DataFrame summary:\n{df_summary}\n\nRelevant context:\n{truncated_context}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred while querying OpenAI: {str(e)}")
        return None

st.title("CSV Analyzer")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read a small sample of the file to check for issues
        sample = uploaded_file.read(1024)
        if not sample:
            st.error("The uploaded file is empty.")
        else:
            # Reset file pointer to the beginning
            uploaded_file.seek(0)
            
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("The uploaded file contains no data.")
            else:
                st.write("Data Preview:")
                st.write(df.head())

                json_data = df.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="converted_data.json",
                    mime="application/json"
                )

                anomalies = analyze_anomalies(df)
                if anomalies:
                    st.subheader("Anomalies Detected:")
                    for anomaly in anomalies:
                        st.write(f"- {anomaly}")
                else:
                    st.write("No anomalies detected.")

                # Chunk the dataframe
                chunks = chunk_dataframe(df)

                # Create a summary of the dataframe
                df_summary = summarize_dataframe(df)

                st.subheader("Analyze your data:")
                user_question = st.text_input("Ask a question about the uploaded file:")
                if user_question:
                    relevant_chunks = get_most_relevant_chunks(user_question, chunks)
                    context = "\n".join([chunk.to_string() for chunk in relevant_chunks])
                    answer = ask_openai(user_question, context, df_summary)
                    if answer:
                        st.markdown("### AI Analysis Report:")
                        st.markdown(answer)
                        
                        with st.expander("View relevant data"):
                            for i, chunk in enumerate(relevant_chunks, 1):
                                st.write(f"Data Subset {i}:")
                                st.write(chunk)

    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or contains no parsable data.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
