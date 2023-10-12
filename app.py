import streamlit as st
from streamlit import session_state as ss
import streamlit_scrollable_textbox as stx
import cv2
import numpy as np
from PIL import Image
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import StreamlitChatMessageHistory
from required_functions import image_to_story,get_image_caption,conv_agent,crawl_iamges,generate_image,image_outpainting

from dotenv import load_dotenv
load_dotenv()
openai.api_type = os.getenv("api_type")
openai.api_base = os.getenv("api_base")
# openai.api_version = os.getenv("api_version")
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model='gpt-4',temperature=0)

if not os.path.exists('./tempDir'):

   # Create a new directory because it does not exist
   os.makedirs('./tempDir')

if not os.path.exists('./tempDir/actual_image'):

   # Create a new directory because it does not exist
   os.makedirs('./tempDir/actual_image')

if not os.path.exists('./tempDir/downloaded_images'):

   # Create a new directory because it does not exist
   os.makedirs('./tempDir/downloaded_images')

if not os.path.exists('./tempDir/generated_images'):

   # Create a new directory because it does not exist
   os.makedirs('./tempDir/generated_images')

if not os.path.exists('./tempDir/editing_images'):

   # Create a new directory because it does not exist
   os.makedirs('./tempDir/editing_images')

img_storage_path = "./tempDir/downloaded_images"
generated_img_path = "./tempDir/generated_images"
edit_img_path = "./tempDir/editing_images"

files1 = os.listdir('./tempDir/actual_image')
for file in files1:
    file_path = os.path.join('./tempDir/actual_image', file)
    if os.path.isfile(file_path):
        os.remove(file_path)

files2 = os.listdir(img_storage_path)
for file in files2:
    file_path = os.path.join(img_storage_path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

files3 = os.listdir(generated_img_path)
for file in files3:
    file_path = os.path.join(generated_img_path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

files4 = os.listdir(edit_img_path)
for file in files4:
    file_path = os.path.join(edit_img_path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir/actual_image",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
    #  return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

st.set_page_config(layout="wide")


def upload_files():
    uploaded_files = st.file_uploader("Upload an image")
    if uploaded_files is not None:
        st.success("Image uploaded successfully")
    else:
        st.error("Please upload an image")

    return uploaded_files
with st.expander("Upload File"):
    uploaded_file = upload_files()

if uploaded_file is not None:
    save_uploadedfile(uploaded_file)
    image_path = "./tempDir/actual_image/"+uploaded_file.name 
    display_image = Image.open(uploaded_file)
    display_image = display_image.resize((300,300))
    @st.cache_data
    def caption_and_story(uploaded_file,image_path):
    #os.path.join("tempDir",uploaded_file.name)
        if "image_caption" not in ss:
            ss["image_caption"] = ""
        if "image_story" not in ss:
            ss["image_story"] = ""
        image_story = image_to_story(image_path)
        image_caption = get_image_caption(image_path).capitalize()
        return image_story,image_caption
    image_story, image_caption = caption_and_story(uploaded_file,image_path)

    _,col1,col2,_ = st.columns((1,10,10,1))

    with col1:
        st.markdown("<h2 style='text-align :center;'> <ins> {} <ins> </h2>".format("The Image"), unsafe_allow_html=True)
        _,col11,_ = st.columns((3,6,3))
        with col11:
            st.image(display_image,channels='BGR')
            st.markdown(f"**<p style = 'text-align:center;'> {ss.get('image_caption')} </p>**",unsafe_allow_html=True)
    with col2:
        st.markdown("<h2 style='text-align :center;'> <ins> {} <ins> </h2>".format("The Story"), unsafe_allow_html=True)
        _,col11,_ = st.columns((3,30,3))
        with col11:
            story_text = ss.get("image_story")
            stx.scrollableTextbox(story_text,height=350)
    st.markdown("----")

    tab_list = ['Ask The Image','Download Image','Generate Image','Outpaint Image']
    tab1,tab2,tab3,tab4 = st.tabs(tab_list)
    st.markdown("""
    <style>

        .stTabs [data-baseweb="tab-list"] {
            gap: 300px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: normal;
            background-color: #F0F2F6;
            border-radius: 8px 8px 4px 4px;
            gap: 5px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 18px;
        font-weight:bold
        }

        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF;
        }

    </style>""", unsafe_allow_html=True)

    with tab1:
        base_prompt = [{"role":"system","content":"You are a helpful assistant"}]
        if "messages" not in ss:
            ss["messages"]=base_prompt

        def show_message(text):
            message_str = [f"{_['role']}:{_['content']}" for _ in ss['messages'][1:]]
            text.text_area("Messages",value = str("\n".join(message_str)),height = 400)
        def clear_text():
            ss['user_input_widget'] = ""
        text = st.empty()
        show_message(text)
        user_query = st.text_input("What is your question?",value="",key="user_input_widget")
        input_str = "||".join([user_query,image_path])
        msgs = StreamlitChatMessageHistory()
        print(msgs.messages)
        conversational_memory = ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=msgs, k=2, return_messages=True)
        prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=f"""You are an image analyzer. You should provide answers to the questions asked about the image. Use tools only when necessary. 
                              Do not use any tool if you are not sure about the answer.
                              If you don't know the answer, reply with I don't know. 
                              Answer truthfully , and to the best of your knowldege.
                              Think step by step before answering."""),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{user_query}")

        ])

        if st.button("Send"):
            with st.spinner("Generating response..."):
                ss["messages"] +=[{"role":"user","content":user_query}]
                message_response = conv_agent(prompt,conversational_memory,user_query,input_str)
                ss["messages"]+=[{"role":"system","content":message_response}]
                show_message(text)
        if st.button("Clear",on_click=clear_text):
            msgs.clear()
            msgs.add_ai_message("How can I help you?")
            ss["messages"] = base_prompt
            show_message(text)
    
    with tab2:
        crawl_image_form = st.form(key='crawl_image_form')
        user_query = crawl_image_form.text_input("Do you want to specify anything else about the images to be downloaded? If not, keep this space blank and submit","")
        crawl_image_submit = crawl_image_form.form_submit_button('Submit')
        if crawl_image_submit:
            crawl_iamges(image_path,user_query,img_storage_path)
            with st.container():
                _,downloaded_img_col1, downloaded_img_col2,_ = st.columns(4)
            with st.container():
                _,downloaded_img_col3, downloaded_img_col4,_ = st.columns(4)
            downloaded_img_col_lst = [downloaded_img_col1,downloaded_img_col2,downloaded_img_col3,downloaded_img_col4]
            downloaded_img_files = os.listdir(img_storage_path)
            for i,file in enumerate(downloaded_img_files):
                downloaded_img_file_path = os.path.join(img_storage_path, file)
                with downloaded_img_col_lst[i]:
                    st.image(downloaded_img_file_path)
        with tab3:
            generated_image_form = st.form(key='generated_image_form')
            user_query = generated_image_form.text_input("Do you want to specify anything else about the images to be downloaded? If not, keep this space blank and submit","")
            size_option = generated_image_form.selectbox("Please select the size of the image you want to generate",("256x256","512x512","1024x1024"))
            generated_image_submit = generated_image_form.form_submit_button('Submit')
            if generated_image_submit:
                generate_image(image_path,user_query,size_option,generated_img_path)
                with st.container():
                    _,generated_img_col1, generated_img_col2,_ = st.columns((1,10,10,1))
                with st.container():
                    _,generated_img_col3, generated_img_col4,_ = st.columns((1,10,10,1))
                generated_img_col_lst = [generated_img_col1,generated_img_col2,generated_img_col3,generated_img_col4]
                generated_img_files = os.listdir(generated_img_path)
                for i,file in enumerate(generated_img_files):
                    generated_img_file_path = os.path.join(generated_img_path, file)
                    with generated_img_col_lst[i]:
                        st.image(generated_img_file_path)
        with tab4:
            outpainted_image_form = st.form(key="outpainted_image_form")
            user_query = outpainted_image_form.text_input("Please specify what you want to add","")
            size_option = outpainted_image_form.selectbox("Please select the size of the image you want to generate",("256x256","512x512","1024x1024"))
            outpainted_image_submit = outpainted_image_form.form_submit_button('Submit')
            if outpainted_image_submit:
                image_outpainting(image_path,user_query,edit_img_path,size_option)
                st.image(edit_img_path+"/outpainted.png")




