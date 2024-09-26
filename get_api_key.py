import tkinter as tk
from tkinter import filedialog
first_line = ""

# Create a file selector and read the first line from the selected file
def select_file_and_read_first_line():
    global first_line
    # Initialize tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file selection dialog
    file_path = filedialog.askopenfilename(
        title="Select a file", 
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )

    # Check if a file was selected
    if file_path:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
        return first_line
    else:
        print("No file selected")

    

# Run the file selection and reading function
