import streamlit as st
import replicate
import os
import pdfplumber
import pandas as pd
import json
from transformers import AutoTokenizer
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

#file that stores conversations and restores them
CONVERSATION_FILE = "conversations.json"

# App title
st.set_page_config(page_title="Snowflake Arctic")

def main():
    """Main function to initialize and run the app."""
    init_chat_history()
    add_custom_styles()
    add_logo()
    display_sidebar_ui()
    display_chat_messages()
    display_file_content()
    get_and_process_prompt()
    summarize_chat()
    generate_summary_report()

def add_custom_styles():
    """Custom Styling for the app's UI"""
    st.markdown(
        """
        <style>
        root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}

        body, .main {
            margin: 0;
            padding: 0;
        }

        .main {
            background: linear-gradient(to bottom, #ff4d4d, #ffffff);
            color: black;
            margin: 0;
        }

        .stSidebar {
            background: linear-gradient(to bottom, #ff4d4d, #ffffff);
            color: black;
        }

        .fixed-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: #C91344;
            z-index: 999;
            border-bottom: 3px solid black;
            padding: 10px 0;
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .fixed-header h2 {
            margin: 0;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
        }

        .content {
            margin-top: 80px;
            padding-left: 20px;
            padding-right: 20px;
        }

        .st-chat-container {
            margin: 20px 0;
            padding: 10px;
        }

        .st-chat-message {
            margin: 10px 0;
            padding: 15px;
            border-radius: 15px;
            font-size: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease;
        }

        .st-chat-message.user {
            background-color: #f0f0f0;
            color: black;
            text-align: right;
            margin-left: auto;
            border-radius: 15px 15px 0 15px;
        }

        .st-chat-message.assistant {
            background-color: #ff4d4d;
            color: white;
            text-align: left;
            margin-right: auto;
            border-radius: 15px 0 15px 15px;
        }

        .st-chat-message:hover {
            background-color: #ffe6e6;
        }

        .st-chat-input input {
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #ff4d4d;
            font-size: 90px;
        }

        .stButton > button {
            background-color: #ff0000;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            font-size: 16px;
            border-radius: 12px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
            cursor: pointer;
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #cc0000;
            box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.3);
        }

        .css-1d391kg {
            width: 350px;
        }

        .st-chat-input {
            margin-top: 20px;
            margin-bottom: 30px;
        }

        .st-chat-container {
            padding: 0 20px;
        }
        
        </style>
        """, unsafe_allow_html=True
    )

def handle_file_upload():
    """Handles file uploads and extracts content for processing."""
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["txt", "pdf", "csv", "json"])
    
    if uploaded_file is not None:
        st.sidebar.write("Filename:", uploaded_file.name)
        st.sidebar.write("File type:", uploaded_file.type)
        st.sidebar.write("File size:", uploaded_file.size)

        extracted_text = ""

        if uploaded_file.type == "text/plain":
            extracted_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        extracted_text += page.extract_text()
            except Exception as e:
                extracted_text = f"Error extracting text from PDF: {e}"
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            extracted_text = df.to_string()
        elif uploaded_file.type == "application/json":
            data = json.load(uploaded_file)
            extracted_text = json.dumps(data, indent=2)

        st.session_state['extracted_text'] = extracted_text
        st.session_state['file_type'] = uploaded_file.type

def display_file_content():
    """Displays extracted file content on the main app interface."""
    if 'extracted_text' in st.session_state:
        file_content = st.session_state['extracted_text']
        file_type = st.session_state.get('file_type', '')

        st.write("### Extracted File Content:")
        
        if file_type in ["text/plain", "application/pdf"]:
            st.text(file_content)
        elif file_type == "text/csv":
            df = pd.read_csv(pd.compat.StringIO(file_content))
            st.dataframe(df)
        elif file_type == "application/json":
            st.json(json.loads(file_content))

def add_logo():
    """Displays a custom logo and header at the top of the app."""
    st.markdown(
        "<div class='fixed-header'><img src='https://i.ytimg.com/vi/X13SUD8iD-8/maxresdefault.jpg' alt='Logo' style='max-width: 150px; margin-right: 10px;'><h2>Kellogg AI</h2></div>",
        unsafe_allow_html=True
    )

def display_sidebar_ui():
    """Creates the sidebar UI with controls and options."""
    with st.sidebar:
        st.title('Snowflake Arctic')
        st.subheader("Adjust model parameters")
        st.slider('temperature', min_value=0.01, max_value=5.0, value=0.3, step=0.01, key="temperature")
        st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01, key="top_p")
        
        handle_file_upload()
        display_note_taking_area()
        
        st.button('Clear chat history', on_click=clear_chat_history)
        export_conversation_history_buttons()
        save_current_conversation()
        display_saved_conversations()

def clear_chat_history():
    """Clears the chat history and resets related session state variables."""
    st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm Arctic, an open language model by Snowflake AI. Ask me anything."}]
    st.session_state.chat_aborted = False
    st.session_state.pop('notes', None)
    st.session_state.pop("summary", None)

def init_chat_history():
    """Initializes chat history and related session state variables."""
    if "messages" not in st.session_state:
        clear_chat_history()
    if "notes" not in st.session_state:
        st.session_state.notes = ""
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
        load_conversations()
    if "selected_conversation_key" not in st.session_state:
        st.session_state.selected_conversation_key = None

def display_chat_messages():
    """Displays chat messages in the app interface."""
    icons = {"assistant": "./Snowflake_Logomark_blue.svg", "user": "⛷️"}
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=icons[message["role"]]):
            st.write(message["content"])

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Returns a tokenizer instance to ensure prompt size compliance."""
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Calculates the number of tokens in a given prompt."""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

def abort_chat(error_message: str):
    """Aborts chat and displays an error message."""
    if st.session_state.messages[-1]["role"] != "assistant":
        st.session_state.messages.append({"role": "assistant", "content": f":red[{error_message}]"})
    else:
        st.session_state.messages[-1]["content"] = f":red[{error_message}]"
    st.session_state.chat_aborted = True
    st.rerun()

def get_and_process_prompt():
    """Processes user prompt and generates responses, including file data if available."""
    file_data = st.session_state.get('extracted_text', '')
    if prompt := st.chat_input("Ask a question or use file data for prompt..."):
        combined_prompt = prompt + "\n\nFile Content:\n" + file_data if file_data else prompt
        st.session_state.messages.append({"role": "user", "content": combined_prompt})
        st.rerun()

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="./Snowflake_Logomark_blue.svg"):
            response = generate_arctic_response()
            st.write_stream(response)

    if st.session_state.chat_aborted:
        st.button('Reset chat', on_click=clear_chat_history, key="clear_chat_history")
        st.chat_input(disabled=True)

def generate_arctic_response():
    """Generates a response using Snowflake Arctic."""
    prompt = []
    for dict_message in st.session_state.messages:
        role_text = "<|im_start|>user" if dict_message["role"] == "user" else "<|im_start|>assistant"
        prompt.append(f"{role_text}\n{dict_message['content']}<|im_end|>")
    prompt.append("<|im_start|>assistant")

    prompt_str = "\n".join(prompt)
    num_tokens = get_num_tokens(prompt_str)
    max_tokens = 2000

    if num_tokens >= max_tokens:
        abort_chat(f"Conversation length too long. Please keep it under {max_tokens} tokens.")

    st.session_state.messages.append({"role": "assistant", "content": ""})
    for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                                  input={"prompt": prompt_str, "temperature": st.session_state.temperature, "top_p": st.session_state.top_p}):
        st.session_state.messages[-1]["content"] += str(event)
        yield str(event)

def summarize_chat():
    """Summarizes the entire chat history."""
    if st.button("Summarize Chat"):
        chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
        summary_prompt = (
            f"Summarize the following conversation into a concise paragraph:\n\n{chat_history}\n\nSummary:"
        )
        summary = replicate.run(
            "snowflake/snowflake-arctic-instruct",
            input={"prompt": summary_prompt, "temperature": 0.5, "top_p": 0.9}
        )
        st.session_state['summary'] = f"### Chat Summary\n\n{summary}"
        st.write(st.session_state['summary'])

def generate_summary_report():
    """Generates and exports a summary report of the chat as a PDF."""
    if 'summary' in st.session_state:
        summary = st.session_state['summary']
        st.write("### Data Report:")
        st.write(summary)

        if st.button('Export Summary Report as PDF'):
            export_summary_pdf(summary)

def export_summary_pdf(summary):
    """Exports the chat summary as a PDF."""
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, "Chat Summary Report")
    p.drawString(100, 720, "Summary:")
    text_object = p.beginText(100, 700)
    text_object.textLines(summary)
    p.drawText(text_object)
    p.save()
    buffer.seek(0)
    st.download_button("Download Report", buffer, "summary_report.pdf", "application/pdf")

def display_note_taking_area():
    """Displays a note-taking section in the sidebar."""
    st.sidebar.subheader("Notes")
    notes = st.sidebar.text_area("Type your notes here...", height=200, key="notes")
    st.session_state[notes] = notes

    if st.sidebar.button("Download Notes as PDF"):
        if notes:
            export_notes_pdf(notes)
        else:
            st.sidebar.warning("Note section is empty. Add some notes first.")

def export_notes_pdf(notes):
    """Exports notes as a PDF."""
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, "Notes")
    text_object = p.beginText(100, 720)
    text_object.textLines(notes)
    p.drawText(text_object)
    p.save()
    buffer.seek(0)
    st.sidebar.download_button("Download Notes", buffer, "notes.pdf", "application/pdf")

def export_conversation_history_buttons():
    """Displays buttons for exporting chat history."""
    if st.sidebar.button("Export as .txt"):
        export_history_as_txt()
    if st.sidebar.button("Export as .json"):
        export_history_as_json()

def export_history_as_txt():
    """Exports chat history as a text file."""
    conversation = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
    notes = st.session_state.get("notes", "")
    content = f"Conversation History:\n\n{conversation}\n\nNotes:\n\n{notes}"

    st.sidebar.download_button("Download Conversation and Notes as .txt", content, "conversation_history.txt", "text/plain")

def export_history_as_json():
    """Exports chat history as a JSON file."""
    conversation = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
    notes = st.session_state.get("notes", "")
    data = {"conversation": conversation, "notes": notes}

    json_data = json.dumps(data, indent=4)
    st.sidebar.download_button("Download Conversation and Notes as .json", json_data, "conversation_history.json", "application/json")

def load_conversations():
    """Loads saved conversations from the file."""
    if not os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "w") as f:
            json.dump({}, f)
    try:
        with open(CONVERSATION_FILE, "r") as f:
            st.session_state.conversations = json.load(f)
    except json.JSONDecodeError:
        st.session_state.conversations = {}

def save_conversation_to_file():
    """Saves the current conversation to the file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_data = {
        "messages": st.session_state.messages,
        "notes": st.session_state.get("notes", ""),
        "summary": st.session_state.get("summary", "")
    }
    st.session_state.conversations[timestamp] = conversation_data
    with open(CONVERSATION_FILE, "w") as f:
        json.dump(st.session_state.conversations, f, indent=4)

def save_current_conversation():
    """Displays a button to save the current conversation."""
    if st.sidebar.button("Save Conversation"):
        save_conversation_to_file()
        st.sidebar.success("Conversation saved!")

def display_saved_conversations():
    """
    Displays a dropdown to select saved conversations
    and provides options to delete or clear conversations.
    """
    st.sidebar.subheader("Saved Conversations")
    conversations = st.session_state.conversations

    if conversations:
        selected_key = st.sidebar.selectbox(
            "Select a saved conversation",
            options=["Select"] + list(conversations.keys()),
            index=0
        )

        if selected_key and selected_key != "Select":
            st.session_state.selected_conversation_key = selected_key
            show_selected_conversation(selected_key)
        if st.sidebar.button("Clear All Conversations"):
            st.session_state.conversations.clear()
            with open(CONVERSATION_FILE, "w") as f:
                json.dump({}, f)
            st.sidebar.success("All conversations cleared!")
            st.experimental_rerun()
    else:
        st.sidebar.info("No saved conversations available.")

def show_selected_conversation(selected_key):
    """
    Displays the details of the selected conversation,
    including chat messages, notes, and summary.
    """
    selected_conversation = st.session_state.conversations[selected_key]
    st.write(f"### Conversation from {selected_key}")
    st.write("#### Chat Messages")
    for msg in selected_conversation["messages"]:
        role = "User" if msg["role"] == "user" else "Assistant"
        st.write(f"**{role}**: {msg['content']}")

    st.write("#### Notes")
    notes = selected_conversation.get("notes", "")
    if notes:
        st.text_area("Conversation Notes", notes, height=200, key=f"notes_{selected_key}", disabled=True)
    else:
        st.info("No notes for this conversation.")

    st.write("#### Summary")
    summary = selected_conversation.get("summary", "")
    if summary:
        st.write(summary)
    else:
        st.info("No summary available for this conversation.")

    if st.button(f"Delete Conversation ({selected_key})"):
        del st.session_state.conversations[selected_key]
        with open(CONVERSATION_FILE, "w") as f:
            json.dump(st.session_state.conversations, f, indent=4)
        st.success(f"Conversation '{selected_key}' deleted!")
        st.experimental_rerun()


if __name__ == "__main__":
    main()
