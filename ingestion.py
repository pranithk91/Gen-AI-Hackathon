import csv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


sv_file_path = 'properties.csv'
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