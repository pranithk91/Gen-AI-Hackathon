import streamlit as st

# Title of the app
st.title("API Key Upload")

# File uploader for text files

def api_key_upload():
    uploaded_file = st.file_uploader("Choose a text file", type=["txt"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Read the file as string
        first_line = uploaded_file.readline().decode("utf-8").strip()  # Read the first line and decode it
        st.write(f"First line of the file: {first_line}")  # Display the first line

        # Simulate closing the app by stopping further execution
        st.write("File uploaded successfully. Closing the app...")
        return first_line
        # Stop the app after file upload and reading the first line
        st.stop()

    # If no file is uploaded, show the upload prompt
    else:
        st.write("Please upload a .txt file to read the first line.")