import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from taipy.gui import Gui, notify

# API Key
os.environ["OPENAI_API_KEY"] = "sk-svcacct-e1Mf-ln-AV_N9D6hVR0eLRbt2b368J8GV7UHq99g-8pWZx5pxo5JgsFVyDzuFScT3BlbkFJ8N33U30wejnMAuDOm-3fguAcc4nL59ciLYH2kUxDGMw71aRruaU5NEEUjl_OD5QA"

#client = OpenAI(api_key="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")

# Sample property data
sample_properties = [
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

# Convert sample data to documents
documents = [
    Document(
        page_content=f"Property ID: {prop['id']}\nType: {prop['type']}\nBedrooms: {prop['bedrooms']}\n"
                     f"Bathrooms: {prop['bathrooms']}\nPrice: ${prop['price']}\nLocation: {prop['location']}\n"
                     f"Features: {prop['features']}",
        metadata={"source": f"property_{prop['id']}"}
    ) for prop in sample_properties
]

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Set up the language model
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

# Set up the conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Function to interact with the bot
def chat_with_bot(query):
    result = qa_chain({"question": query})
    return result["answer"]


user_input = ""
chat_history = "Property Recommendation Bot: Hello! I can help you find properties. What kind of property are you looking for?\n\n"

page = """
<|layout|columns=340px 1fr|
<|part|render=True|
# Property Recommendation Bot
<|{user_input}|input|label=Enter your question:|>
<|Ask|button|on_action=ask_bot|>
|>
<|part|render=True|
<|{chat_history}|markdown|height=500px|
|>
|>
"""

def ask_bot(state):
    if state.user_input:
        response = chat_with_bot(state.user_input)
        state.chat_history += f"You: {state.user_input}\n\n"
        state.chat_history += f"Property Recommendation Bot: {response}\n\n"
        state.user_input = ""  # Clear the input field
        notify(state, "info", "Bot has responded!")
    else:
        notify(state, "warning", "Please enter a question!")

if __name__ == "__main__":
    gui = Gui(page)
    gui.run(use_reloader=True)