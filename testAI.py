import os
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget
import pandas as pd
import csv
# Import LangChain-related modules
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Function to read API key from file
from get_api_key import select_file_and_read_first_line

# Set OpenAI API key
api_key = select_file_and_read_first_line()
os.environ["OPENAI_API_KEY"] = api_key

# Sample property data
# Function to convert CSV to a list of dictionaries
def csv_to_dict_list(csv_file_path):
    with open(csv_file_path, mode='r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)  # Read rows into dictionaries
        data_list = [dict(row) for row in reader]  # Convert each row into a dictionary
    return data_list


csv_file_path = 'properties.csv'
sample_properties = csv_to_dict_list(csv_file_path)


# Convert sample data to documents

documents = [
    Document(
        page_content=f"Property ID: {prop['Name']}\nYear Built: {prop['Year Built']}\nBedrooms: {prop['Bedrooms']}\n"
                     f"Bathrooms: {prop['Bathrooms']}\nRent: ${prop['Rent']}\nLocation: {prop['Zipcode']}\n"
                     f"Street: {prop['Street']}\nCity: {prop['City']}\nState:{prop['State']}\n"
                     f"County: {prop['County']}\nPool: {prop['Swimming Pool']}\nSize: ${prop['Square Feet']}\nLeasable: {prop['Leasable']}",
        metadata={"source": f"property_{prop['Name']}"}
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

# PyQt6 chatbot interface
class ChatbotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Property Recommendation Chat Bot")
        self.setGeometry(300, 300, 600, 400)

        # Main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout for the chat interface
        layout = QVBoxLayout()

        # Text area for the conversation
        self.chat_area = QTextEdit(self)
        self.chat_area.setReadOnly(True)
        layout.addWidget(self.chat_area)

        # Input field for user to type their message
        self.input_field = QLineEdit(self)
        layout.addWidget(self.input_field)

        # Submit button
        self.submit_button = QPushButton("Send", self)
        layout.addWidget(self.submit_button)

        # Set layout to the central widget
        self.central_widget.setLayout(layout)

        # Connect button click to the bot response
        self.submit_button.clicked.connect(self.handle_user_input)

    def handle_user_input(self):
        # Get user input
        user_input = self.input_field.text()

        # Add user input to chat area
        self.chat_area.append(f"You: {user_input}")

        # Get the bot's response based on the input
        bot_response = chat_with_bot(user_input)

        # Add bot response to chat area
        self.chat_area.append(f"Bot: {bot_response}\n")

        # Clear the input field for the next message
        self.input_field.clear()

# Main application loop
def main():
    app = QApplication(sys.argv)
    chatbot_window = ChatbotWindow()
    chatbot_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
