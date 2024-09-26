# app.py
import os
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import taipy as tp
from taipy.gui import Gui, notify

from get_api_key import select_file_and_read_first_line, first_line



api_key = select_file_and_read_first_line()




os.environ["OPENAI_API_KEY"] = api_key

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
    search_results = faiss_index.similarity_search(user_query, k=3)  # Top 3 results
    results_df = data_frame.iloc[[result["index"] for result in search_results]]
    
    properties_str = results_df.to_string(index=False)  # Convert DataFrame to readable format
    
    # Create LLM Chain
    llm = OpenAI(model="text-davinci-003", temperature=0.7)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Generate response with recommendations
    response = chain.run(user_preferences=user_query, properties=properties_str)
    return results_df  # Return the DataFrame for display in Taipy

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

# State variables to hold the user prompt and recommendations
user_prompt = ""
recommendations = []

# Function triggered when the user submits their prompt
def on_user_submit(state):
    global recommendations
    if not state.user_prompt:
        notify(state, "error", "Please enter a valid property preference.")
    else:
        recommendations = get_property_recommendations(state.user_prompt, df, faiss_index).to_dict(orient='records')
        notify(state, "success", "Recommendations updated.")

# Taipy GUI layout with user input and recommendation display
page = """
# Property Recommendation Bot

## Enter your preferences for a property:

<|{user_prompt}|input|label=User Preferences|>

<|Submit|button|on_action=on_user_submit|>

## Recommended Properties:

<|for rec in recommendations|>

<|card
### {rec['location']}
- Price: {rec['price']} USD
- Bedrooms: {rec['bedrooms']}
- Type: {rec['type']}
|>

<|endfor|>
"""

# Run the Taipy app
if __name__ == "__main__":
    Gui(page).run()