import streamlit as st
import pandas as pd
#import plotly.express as px
import numpy as np

st.set_page_config(page_title="ESG Interactive AI Dashboard", page_icon=":earth_asia:",layout="wide")
st.title('🌏 ESG Interactive AI Dashboard')

def load_data():
    df = pd.read_csv('data/Data Greenie All - Sheet1.csv')
    return df

data = load_data()
st.title('🚀 Welcome to ESG Investments Dashboard with Gemini AI-Chatbot Analysis')

welcome_text = (
    "In today's financial world, it's becoming more important to consider "
    "Environmental, Social, and Governance (ESG) factors when investing. "
    "Our team 'Greenie' created a special tool called an ESG Investments Dashboard "
    "with Gemini AI-Chatbot Analysis. This dashboard brings together important "
    "information about how companies are doing in terms of ESG, leveraging our "
    "Gemini AI Chatbot for analysis. Utilizing the technology such as Natural "
    "Language Processing (NLP) and Data Visualization, we aim for user-friendly "
    "comprehension. With this dashboard, investors can make better choices that "
    "match their goals for a better world."
    "This project is developed by Greenie Team for QuantCorner-LSEG Hackathon: the Quest for Sustainable Alpha"
)

st.write(welcome_text)
st.write('—' * 80)
# Display the data using Streamlit
#st.title('ESG Data')
#st.write('Here is the imported data:')
#st.dataframe(data)

def load_data():
    df = pd.read_csv('data/Data Greenie All - Sheet1.csv') 
    return df 
df = load_data()
#sidebar
st.sidebar.title("Parameters")

S = st.sidebar.selectbox('Select a Stock name', df['Stock Name']), 
                     

X = st.sidebar.selectbox('Sector', df['Sector'])

# Filter data
filtered_df1 = df[df['Stock Name'] == S]
filtered_df2 = df[df['Sector'] == X]
# Display the filtered data
#st.write(filtered_df1,filtered_df2)
#data.drop('index',axis=1)
#options = ['Environment', 'Social', 'Governance']

#selected_option = st.selectbox('Select an ESG Pillar Score:', options)

#st.write('You selected:', selected_option)

from PIL import Image
import io
#import os
import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm

with st.sidebar:
    st.title("Gemini API")
    api_key = st.text_input("API key")
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key
     #   genai.configure(api_key=api_key)
    #else:
     #   if "api_key" in st.secrets:
    #        genai.configure(api_key=st.secrets["api_key"])
     #       st.success('API key already provided!', icon='✅')
      #      api_key = st.secrets['GOOGLE_API_KEY']
       # else:
        #   api_key = st.text_input('Enter Google API Key:', type='password')
         #   if not (api_key.startswith('AI')):
         #     st.warning('Please enter your API Key!', icon='⚠️')
          #  else:
           #   st.success('Success!', icon='✅')
    
    "[Get a Google Gemini API key](https://ai.google.dev/)"
    "[View the source code](https://github.com/wms31/streamlit-gemini/blob/main/app.py)"
    "[Check out the blog post!](https://letsaiml.com/creating-google-gemini-app-with-streamlit/)"

    select_model =["gemini-pro-vision"]
    if select_model == "gemini-pro-vision":
        uploaded_image = st.file_uploader(
            "upload image",
            label_visibility="collapsed",
            accept_multiple_files=False,
            type=["png", "jpg"],
        )
        st.caption(
            "Note: The vision model gemini-pro-vision is not optimized for multi-turn chat."
        )
        if uploaded_image:
            image_bytes = uploaded_image.read()

def get_response(messages, model="gemini-pro"):
    model = genai.GenerativeModel(model)
    res = model.generate_content(messages, stream=True,
                                safety_settings={'HARASSMENT':'block_none'})
    return res

if "messages" not in st.session_state:
    st.session_state["messages"] = []
messages = st.session_state["messages"]

# The vision model gemini-pro-vision is not optimized for multi-turn chat.
#if messages and select_model != "gemini-pro-vision":
 #   for item in messages:
  #      role, parts = item.values()
   #     if role == "user":
    #        st.chat_message("user").markdown(parts[0])
     #   elif role == "model":
      #      st.chat_message("assistant").markdown(parts[0])

#chat_message = st.chat_input("Say something")
#generate_t2t = st.button("Generate my travel itinerary", key="generate_t2t")

with st.container():
  destination_name = st.text_input("Stock Name: \n\n",value="MINT.BK")
  days = st.text_input("Average of ESG Score",value="89.815")
  suggested_attraction = st.text_input("Total Return",value="0.8547")
  config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
        }
    
  generate_t2t = st.button("Generate", key="generate_t2t")
  model = genai.GenerativeModel("gemini-pro", generation_config=config)
  if generate_t2t:
      with st.spinner("Generating your travel itinerary using Gemini..."):
          response = model.generate_content("Please analyst ESG")
          st.write("Your plan:")
          st.write(response.text)
