import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import re

# Set up OpenAI client
client = OpenAI(api_key=st.Secrets("OPENAI_API_KEY"))

def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    return df

def analyze_query(query, df, column_instruction):
    query_types = {
        "Forwarding Email": ["User agent", "Request ID", "User type", "Cross tenant access type"],
        "Suspicious User Password Change": ["Date (UTC)", "Request ID", "Correlation ID", "User ID", "User", "Username", "User type", "Cross tenant access type", "Incoming token type", "Flagged for review", "Token issuer type", "Conditional Access", "Federated Token Id", "Federated Token Issuer"],
        "User accounts added or Deleted": ["User", "Username", "User type", "User ID", "Request ID", "Correlation ID", "Cross tenant access type", "Token issuer type", "Token issuer name", "Associated Resource Id", "Federated Token Id", "Federated Token Issuer"],
        "Audit Logs Disabled": ["Flagged for review", "Token issuer type", "Token issuer name", "Conditional Access", "Federated Token Id", "Federated Token Issuer"],
        "MFA disabled": ["Conditional Access", "Token issuer type"],
        "Record Type Based alerts": ["Flagged for review", "Token issuer type", "Incoming token type", "Token issuer name", "Conditional Access", "Managed Identity type"],
        "Device No Longer Compliant": ["User agent", "Cross tenant access type", "Incoming token type", "Incoming token type.1", "Conditional Access", "Managed Identity type", "Associated Resource Id"],
        "Suspicious Inbox Manipulation Rule": ["Flagged for review", "Conditional Access", "Cross tenant access type", "Incoming token type", "Token issuer type", "Latency", "Federated Token Id", "Federated Token Issuer"],
        "Insight and report events": ["Flagged for review", "Latency", "Conditional Access", "Managed Identity type", "Federated Token Id", "Federated Token Issuer"],
        "EOP Phishing and Malware events": ["Flagged for review", "User agent", "Correlation ID", "User ID", "Username", "Token issuer type", "Token issuer name", "Federated Token Id", "Federated Token Issuer"],
        "Member added to Group": ["User ID", "User", "Username", "User type", "Request ID", "Correlation ID", "Incoming token type", "Token issuer type", "Associated Resource Id", "Cross tenant access type"],
        "Member added to Role": ["User", "Username", "User type", "Cross tenant access type", "Incoming token type", "Token issuer type", "Conditional Access", "Associated Resource Id", "Federated Token Id", "Federated Token Issuer"],
        "Unusual amount of login failures": ["Request ID", "User ID", "Username", "User type", "Cross tenant access type", "Flagged for review", "Conditional Access", "Token issuer type", "Token issuer name", "Latency"],
        "Possible Brute Force Lockout Evasion": ["Request ID", "User agent", "User ID", "Username", "User type", "Cross tenant access type", "Incoming token type", "Flagged for review", "Token issuer type", "Conditional Access", "Managed Identity type", "Associated Resource Id", "Federated Token Id", "Federated Token Issuer"],
        "Impossible Travel Alerts": ["Date (UTC)", "User agent", "Correlation ID", "User ID", "User", "Username", "User type", "Cross tenant access type", "Incoming token type", "Conditional Access", "Latency", "Associated Resource Id", "Token issuer type", "Token issuer name", "Federated Token Id", "Federated Token Issuer"],
        "Sign ins with Blacklisted IPs": ["User agent", "User ID", "User", "Username", "User type", "Cross tenant access type", "Incoming token type", "Conditional Access", "Federated Token Id", "Federated Token Issuer", "Token issuer type", "Token issuer name", "Associated Resource Id", "Flagged for review"],
        "Sign ins with anonymous IPs": ["User agent", "User ID", "User type", "Incoming token type", "Incoming token type.1", "Token issuer type", "Federated Token Id", "Federated Token Issuer"],
        "Foreign country alerts": ["Correlation ID", "User agent", "User type", "Cross tenant access type", "Incoming token type", "Flagged for review", "Token issuer type", "Token issuer name", "Incoming token type.1", "Latency", "Conditional Access", "Managed Identity type", "Associated Resource Id", "Federated Token Id", "Federated Token Issuer"],
        "Unusual logins": ["Date (UTC)", "User agent", "User ID", "User", "Username", "User type", "Cross tenant access type", "Incoming token type", "Flagged for review", "Token issuer type", "Latency", "Conditional Access", "Managed Identity type"]
    }

    # Check if any query type keyword is in the query
    query_type = next((qt for qt in query_types if qt.lower() in query.lower()), None)

    if query_type:
        relevant_columns = query_types[query_type]
    else:
        # Use GPT to determine relevant columns
        messages = [
            {"role": "system", "content": column_instruction.format(columns=', '.join(df.columns))},
            {"role": "user", "content": query}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=16384,
            temperature=0.1
        )
        
        relevant_columns = [col.strip() for col in response.choices[0].message.content.split(',')]
    
    return [col for col in relevant_columns if col in df.columns], query_type

def get_ai_response(prompt, data_description, relevant_data, query_type, df, response_instruction):
    # Prepare a sample of the entire dataset (first 100 rows)
    full_data_sample = df.head(100).to_string()

    messages = [
        {"role": "system", "content": response_instruction.format(full_data_sample=full_data_sample, columns=', '.join(df.columns))},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=16384,
        temperature=0.1
    )
    
    return response.choices[0].message.content

def main():
    st.title("Log Analysis Chatbot")
    
    if not client.api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return

    # Default column instructions
    default_column_instruction = """You are a data analyst. Analyze the following query and determine which columns from the dataset are relevant. The available columns are: {columns}. Respond with only the column names, separated by commas."""

    # Default response instructions
    default_response_instruction = """You are an advanced AI assistant specialized in analyzing security log data. Your task is to provide comprehensive and accurate responses to queries about the security logs. Follow these instructions:

1. Thoroughly analyze the entire dataset to find all relevant information.
2. Provide responses based on related columns, rows, and contents from across the whole document.
3. Ensure no relevant content is missed in your response.
4. If the answer requires information from multiple parts of the document, combine and synthesize this information.
5. If you need more specific data or information about certain rows that are not in the provided sample, please mention this in your response.
6. Do not give irrelevant or fake responses.
7. Give the response as a report.

Here's a sample of the dataset (first 100 rows):
{{full_data_sample}}

The full set of columns available in the dataset are: {{columns}}

Remember, your goal is to provide the most comprehensive and accurate response possible based on the entire security log dataset."""

    # Use session state to store instructions
    if 'column_instruction' not in st.session_state:
        st.session_state.column_instruction = default_column_instruction
    if 'response_instruction' not in st.session_state:
        st.session_state.response_instruction = default_response_instruction
    if 'temp_column_instruction' not in st.session_state:
        st.session_state.temp_column_instruction = st.session_state.column_instruction
    if 'temp_response_instruction' not in st.session_state:
        st.session_state.temp_response_instruction = st.session_state.response_instruction

    # Create checkboxes to select which instructions to edit
    col1, col2 = st.columns(2)
    edit_column_instructions = col1.checkbox("Edit Column Instructions")
    edit_response_instructions = col2.checkbox("Edit Response Instructions")

    # Create text areas for editing instructions
    if edit_column_instructions:
        st.session_state.temp_column_instruction = st.text_area(
            "Edit Column Instructions", 
            value=st.session_state.temp_column_instruction, 
            height=200
        )
        if st.button("Save Column Instructions"):
            st.session_state.column_instruction = st.session_state.temp_column_instruction
            st.success("Column Instructions saved successfully!")

    if edit_response_instructions:
        st.session_state.temp_response_instruction = st.text_area(
            "Edit Response Instructions", 
            value=st.session_state.temp_response_instruction, 
            height=300
        )
        if st.button("Save Response Instructions"):
            st.session_state.response_instruction = st.session_state.temp_response_instruction
            st.success("Response Instructions saved successfully!")

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            data_description = f"Number of rows: {len(df)}\n"
            data_description += f"Data types: {df.dtypes.to_string()}"
            
            user_question = st.text_input("Ask a question about the security logs:")
            
            if user_question:
                with st.spinner("Analyzing query..."):
                    relevant_columns, query_type = analyze_query(user_question, df, st.session_state.column_instruction)
                
                if query_type:
                    st.write("Identified Query Type:", query_type)
                st.write("Initially Relevant columns:", relevant_columns)
                
                if relevant_columns:
                    relevant_data = df[relevant_columns].to_string()
                else:
                    relevant_data = "No specific columns identified as initially relevant."
                
                with st.spinner("Generating comprehensive response..."):
                    ai_response = get_ai_response(user_question, data_description, relevant_data, query_type, df, st.session_state.response_instruction)
                
                st.write("AI Response:")
                st.write(ai_response)

if __name__ == "__main__":
    main()
