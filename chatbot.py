import streamlit as st
from tensorflow.keras.models import load_model
from gemma.models import GemmaCausalLM
from custom_obj import GemmaCausalLM
import gdown
import os

custom_objects = {"GemmaCausalLM": GemmaCausalLM}

file_id = "1QGsrTN4GYa5rOr74ClYTrCnamTW-ZmsD"
model_path = "models/gemma_lm_1k.h5"

model = load_model(model_path, custom_objects=custom_objects)

def download_model():
    url = f"https://drive.google.com/uc?id={file_id}"
    output = model_path
    gdown.download(url, output, quiet=False)

# Check if the model exists locally, otherwise download it
if not os.path.exists(model_path):
    st.info("Downloading model. This may take a while...")
    download_model()


st.title("Counsellot: Your counselling bot")


st.write("Ask questions related to your mental health")

if "history" not in st.session_state:
    st.session_state.history = []
if "user_name" not in st.session_state:
    st.session_state.user_name = ""


if not st.session_state.user_name:
    st.session_state.user_name = st.text_input("Enter your name: ", key = "name_input")

    if st.session_state.user_name:
        st.success(f"Hello, {st.session_state.user_name}! Let's talk and cheer you up!")

else:
    st.write(f"Welcome back, {st.session_state.user_name}!")

user_input = st.text_input(f"{st.session_state.user_name}: ", key="input", placeholder = "Type your message...")


# Define a function to generate chatbot responses
def generate_response(input_text):
    
    response = model.generate(input_text)

    return response


if st.button("Send"):
    if user_input:
        response = generate_response(user_input)
        # Convert model's response to text if necessary (e.g., decode or process the response)
        chatbot_reply = response  # Replace this with proper decoding logic if needed
        st.write(f"Bot: {chatbot_reply}")


        st.session_state.history.append({"user": user_input, "bot": response})


# Display chat history
for chat in st.session_state.history:
    st.markdown(f"**{st.session_state.user_name}:** {chat['user']}")
    st.markdown(f"**Counsellot:** {chat['bot']}")




