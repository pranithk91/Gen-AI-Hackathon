import os
import sys
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget
import pandas as pd
import csv
# Import LangChain-related modules
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub

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


#print(documents)
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("Properties")
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

# Set up the conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

vectorstore = FAISS.load_local("Properties", embeddings, allow_dangerous_deserialization=True)
# Create the conversational chain

retrieval_qa_chain_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(
    OpenAI(), retrieval_qa_chain_prompt
)
retrieval_chain = create_retrieval_chain(
    vectorstore.as_retriever(), combine_docs_chain
)
"""qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)"""


# Function to interact with the bot
def chat_with_bot(query):
    result = retrieval_chain.invoke({"input": query})
    print(result)
    return result["answer"]
    

class ChatbotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Property Recommendation Chat Bot")
        self.setGeometry(200, 200, 1000, 800)

        # Main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout for the chat interface
        layout = QVBoxLayout()

        # Text area for the conversation
        self.chat_area = QTextEdit(self)
        self.chat_area.setReadOnly(True)
        layout.addWidget(self.chat_area)

        self.chat_area.setStyleSheet("""
            QTextEdit {
                
                background-image: url('AIEnabled_IMG.png');  opacity: 127;
                background-repeat: no-repeat; 
                background-position: center;
                background-attachment: fixed;
                font-size: 14px;
                padding: 10px;
            }
        """)


        input_layout = QHBoxLayout()

        # Input field for user to type their message
        self.input_field = QLineEdit(self)
        self.input_field.setFixedHeight(60)
        input_layout.addWidget(self.input_field)

        # Submit button
        self.submit_button = QPushButton("Send", self)
        self.submit_button.setFixedSize(50, 30) 
        input_layout.addWidget(self.submit_button)

        layout.addLayout(input_layout)

        # Set layout to the central widget
        self.central_widget.setLayout(layout)

        # Connect button click/Enter key to the bot response
        self.submit_button.clicked.connect(self.handle_user_input)
        self.input_field.returnPressed.connect(self.handle_user_input)

    

    def handle_user_input(self):
        # Get user input
        user_input = self.input_field.text()

        # Add user input to chat area with background color and font size
        self.add_message("You", user_input, user=True)

        # Get the bot's response based on the input
        bot_response = chat_with_bot(user_input)
        #print("Property Recommendation Bot:", bot_response)

        # Add bot response to chat area with background color and font size
        #for content in bot_response['context']:
        self.add_message("Bot", bot_response, user=False)

        

        # Clear the input field for the next message
        self.input_field.clear()

    def add_message(self, sender, message, user=True):
        # HTML styling for the messages
        if user:
            style = "background-color:#D1F2EB; padding:10px; border-radius:10px; font-size:18px; width:70%; margin-left:0;"
            align = "left"
        else:
            style = "background-color:#FADBD8; padding:10px; border-radius:10px; font-size:18px; width:70%; margin-left:auto;"
            align = "right"

        # Add the message to the chat area with HTML formatting
        formatted_message = f'<p style="text-align:{align};"><span style="{style}">{message}</span></p>'
        self.chat_area.append(formatted_message)



# Main application loop
def main():
    app = QApplication(sys.argv)
    chatbot_window = ChatbotWindow()
    chatbot_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
