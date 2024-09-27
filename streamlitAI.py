import streamlit as st
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

from streamlitUpload import api_key_upload
import os


"""api_key = api_key_upload()
st.write(api_key)"""
# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-svcacct-e1Mf-ln-AV_N9D6hVR0eLRbt2b368J8GV7UHq99g-8pWZx5pxo5JgsFVyDzuFScT3BlbkFJ8N33U30wejnMAuDOm-3fguAcc4nL59ciLYH2kUxDGMw71aRruaU5NEEUjl_OD5QA"


# Load sample property dataset
def load_sample_data():
    data = [
  {"id": 1, "type": "House", "bedrooms": 3, "bathrooms": 2, "price": 350000, "location": "New York", "features": "Garden"},
  {"id": 2, "type": "Apartment", "bedrooms": 2, "bathrooms": 1, "price": 250000, "location": "Los Angeles", "features": "Balcony"},
  {"id": 3, "type": "Condo", "bedrooms": 1, "bathrooms": 1, "price": 180000, "location": "Miami", "features": "Pool"},
  {"id": 4, "type": "House", "bedrooms": 4, "bathrooms": 3, "price": 450000, "location": "Miami", "features": "Garage"},
  {"id": 5, "type": "Townhouse", "bedrooms": 3, "bathrooms": 2, "price": 320000, "location": "New York", "features": "Rooftop Terrace"},
  {"id": 6, "type": "House", "bedrooms": 5, "bathrooms": 4, "price": 550000, "location": "Los Angeles", "features": "Fireplace"},
  {"id": 7, "type": "Apartment", "bedrooms": 1, "bathrooms": 1, "price": 200000, "location": "New York", "features": "City View"},
  {"id": 8, "type": "Condo", "bedrooms": 2, "bathrooms": 2, "price": 280000, "location": "Miami", "features": "Gym"},
  {"id": 9, "type": "House", "bedrooms": 3, "bathrooms": 2, "price": 380000, "location": "Los Angeles", "features": "Swimming Pool"},
  {"id": 10, "type": "Townhouse", "bedrooms": 2, "bathrooms": 1, "price": 270000, "location": "New York", "features": "Patio"},
  {"id": 11, "type": "Apartment", "bedrooms": 3, "bathrooms": 2, "price": 300000, "location": "Los Angeles", "features": "Doorman"},
  {"id": 12, "type": "House", "bedrooms": 4, "bathrooms": 3, "price": 420000, "location": "Miami", "features": "Solar Panels"},
  {"id": 13, "type": "Condo", "bedrooms": 1, "bathrooms": 1, "price": 190000, "location": "New York", "features": "Storage Unit"},
  {"id": 14, "type": "House", "bedrooms": 5, "bathrooms": 4, "price": 580000, "location": "Los Angeles", "features": "Home Theater"},
  {"id": 15, "type": "Apartment", "bedrooms": 2, "bathrooms": 1, "price": 230000, "location": "Miami", "features": "Fitness Center"},
  {"id": 16, "type": "Townhouse", "bedrooms": 3, "bathrooms": 2, "price": 340000, "location": "New York", "features": "Community Pool"},
  {"id": 17, "type": "House", "bedrooms": 4, "bathrooms": 3, "price": 470000, "location": "Los Angeles", "features": "Backyard"},
  {"id": 18, "type": "Condo", "bedrooms": 2, "bathrooms": 2, "price": 260000, "location": "Miami", "features": "Concierge"},
  {"id": 19, "type": "Apartment", "bedrooms": 1, "bathrooms": 1, "price": 210000, "location": "New York", "features": "In-unit Laundry"},
  {"id": 20, "type": "House", "bedrooms": 3, "bathrooms": 2, "price": 390000, "location": "Los Angeles", "features": "Screened Porch"}
]
    return pd.DataFrame(data)

# Create a Prompt Template
property_template = """
You are a real estate assistant. Based on the user's preferences, recommend the best properties from the dataset. 
User preferences: {user_preferences}

Here are some properties:
{properties}

Recommend the most suitable properties and explain why.
"""

prompt = PromptTemplate(
    input_variables=["user_preferences", "properties"],
    template=property_template,
)

# Function to get property recommendations
def get_property_recommendations(user_query, data_frame, faiss_index):
    # Get FAISS search results
    search_results = faiss_index.similarity_search(user_query, k=3)  # Top 3 results
    # Extract relevant rows using index values from search results
    result_indices = [result['index'] for result in search_results]
    
    # Ensure we have valid indices and extract rows from the DataFrame
    if result_indices:
        results_df = data_frame.iloc[result_indices]
        return results_df.to_dict(orient='records')  # Return results as a list of dictionaries
    else:
        return []  # Return empty list if no results found

# Prepare embeddings for property data
def prepare_embeddings(data_frame):
    embedding_model = OpenAIEmbeddings()
    texts = data_frame.apply(lambda row: f"{row['location']} {row['price']} {row['bedrooms']} bedrooms {row['type']}", axis=1)
    
    # Build FAISS index for fast similarity search
    faiss_index = FAISS.from_texts(texts, embedding_model)
    return faiss_index

# Initialize the property data and FAISS index
df = load_sample_data()
faiss_index = prepare_embeddings(df)

# Streamlit chatbot UI
st.title("Property Recommendation Chat Bot")

# Chat History storage
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field for user to ask for recommendations
user_input = st.text_input("You:", key="user_input")

"""if user_input:
    # Add user input to chat history
    st.session_state.chat_history.append(f"User: {user_input}")

    # Get recommendations based on the user input
    recommendations = get_property_recommendations(user_input, df, faiss_index)

    # Format the chatbot's response
    if recommendations:
        bot_response = "Here are some properties I found for you:\n"
        for rec in recommendations:
            bot_response += f"- **Location**: {rec['location']}, **Price**: {rec['price']} USD, **Bedrooms**: {rec['bedrooms']}, **Type**: {rec['type']}\n"
    else:
        bot_response = "Sorry, I couldn't find any matching properties."

    # Add bot response to chat history
    st.session_state.chat_history.append(f"Bot: {bot_response}")

    # Clear the input field
    st.session_state.user_input = """""

# Display the chat history
for message in st.session_state.chat_history:
    if message.startswith("User:"):
        st.write(f"**{message}**")
    else:
        st.markdown(message)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    recommendations = get_property_recommendations(user_input, df, faiss_index)

    # Format the chatbot's response
    if recommendations:
        bot_response = "Here are some properties I found for you:\n"
        for rec in recommendations:
            bot_response += f"- **Location**: {rec['location']}, **Price**: {rec['price']} USD, **Bedrooms**: {rec['bedrooms']}, **Type**: {rec['type']}\n"
    else:
        bot_response = "Sorry, I couldn't find any matching properties."
    
    response = st.write_stream(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": response})