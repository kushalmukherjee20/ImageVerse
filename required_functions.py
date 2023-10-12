# lagchain openai icrawler timm pydantic==1.10.8 easyocr torch torchvision torchaudio timm nltk PIL streamlit-scrollable-textbox


from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection, BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch
from io import BytesIO, BufferedReader
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("words")
from nltk.corpus import words
import easyocr

from tempfile import NamedTemporaryFile
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import openai

import re
import requests
import string

import ssl
from icrawler.builtin import GoogleImageCrawler
from datetime import date

import streamlit as st
from streamlit import session_state as ss


from dotenv import load_dotenv
load_dotenv()
openai.api_type = os.getenv("api_type")
openai.api_base = os.getenv("api_base")
# openai.api_version = os.getenv("api_version")
openai.api_key = os.getenv("OPENAI_API_KEY")

device = "cpu"


def detect_objects(image_path):
    image = Image.open(image_path).convert('RGB')

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        detections += ' {}'.format(model.config.id2label[int(label)])
        detections += ' {}\n'.format(float(score))

    return detections

class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objects. Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score."

    def _run(self, input_str):
        split_string = input_str.split("||")
        if len(split_string)>1:
            user_query = split_string[0]
            img_path = split_string[1]
        else:
            img_path = split_string[0]
        image = Image.open(img_path).convert('RGB')

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    

def image_ocr(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path, detail = 0)
    full_text = ''
    for result in results:
        full_text += result + ' '
    return full_text

class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = "Use this tool when given an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, input_str):
        split_string = input_str.split("||")
        if len(split_string)>1:
            user_query = split_string[0]
            img_path = split_string[1]
        else:
            img_path = split_string[0]
        image = Image.open(img_path).convert('RGB')
        image_text = image_ocr(img_path)

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)
        final_caption = caption + " " + image_text
        return final_caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    
def get_image_caption(image_path):
    image = Image.open(image_path).convert('RGB')

    model_name = "Salesforce/blip-image-captioning-large"
    device = "cpu"  # cuda

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    inputs = processor(image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(output[0], skip_special_tokens=True)
    ss["caption"] = caption
    caption = ss["caption"]

    return caption


class ImageOCRTool(BaseTool):
    name = "Image OCR"
    description = "Use this tool when user want perform OCR on the given image and use extracted text from the image."
    def _run(self, input_str):
        split_string = input_str.split("||")
        if len(split_string)>1:
            user_query = split_string[0]
            img_path = split_string[1]
        else:
            img_path = split_string[0]
        reader = easyocr.Reader(['en'])
        results = reader.readtext(img_path, detail = 0)
        full_text = ''
        for result in results:
            full_text += result + ' '
        return full_text 
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


def crawl_iamges(image_path,user_query,img_storage_path):
    image_caption = get_image_caption(image_path)
    detected_objects = re.sub('['+string.punctuation+']', '', detect_objects(image_path)).split()
    detected_objects_lst = [detected_object for detected_object in detected_objects if detected_object in words.words()]
    summarized_rqrmnt_response = openai.ChatCompletion.create(
    model='gpt-4', temperature=0,
    messages=[
        {"role": "user", "content":f"""
            You are a Google search expert and you need to search images in Google.
        There are two texts which needs to be summarized in one sentence.
        Texts: {image_caption},and {user_query}. 
        Objects: {detected_objects_lst} are the objects present in the given image
        Query: Create a Google search query using the above texts in one sentence.
        Constraints: 1. The Google search query should be crisp and concise and should contain the keywords.
                    2. The Google search query should be in one sentence.
                    3. Do not change the meaning {image_caption} until specifically asked in {user_query}
                    4. Make sure only the {detected_objects_lst} are present in the images and remove the numbers and list from it

    """}],)
    summarized_rqrmnt = summarized_rqrmnt_response["choices"][0]["message"]["content"]
    
    
    google_crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=2, 
        downloader_threads=4,
        storage={'root_dir': img_storage_path})
    filters = dict(
    size='large',
    license='commercial,modify')
    google_crawler.crawl(
        keyword=summarized_rqrmnt,filters = filters,
        max_num=4,
        )

def generate_image(image_path,user_query,size_option,generated_img_path):
    image_caption = get_image_caption(image_path)
    detected_objects = re.sub('['+string.punctuation+']', '', detect_objects(image_path)).split()
    detected_objects_lst = [detected_object for detected_object in detected_objects if detected_object in words.words()]
    prompt = f"create images similar to {image_caption} and {user_query}. The image should contain {detected_objects_lst}"
    response = openai.Image.create(
        prompt="{}".format(prompt),
        n=4,
        size=size_option,
    )
    names = ["image_1","image_2","image_3","image_4"]
    for i, name in enumerate(names):
        url = response["data"][i]["url"]    
        image = requests.get(url)
        with open(generated_img_path+"/{}.png".format(name), "wb") as f:
            f.write(image.content)

class ImageVQA(BaseTool):
    name = "Visual question answering"
    description = "Use this tool when users asks specific questions about the image. Don't use this for image captioning or object detection."
    def _run(self, input_str):
        split_string = input_str.split("||")
        if len(split_string)>1:
            user_query = split_string[0]
            img_path = split_string[1]
        else:
            img_path = split_string[0]
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        image = Image.open(img_path).convert('RGB')
        inputs = processor(image, user_query, return_tensors="pt")

        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

def visual_qa(image_path,user_query):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    image = Image.open(image_path).convert('RGB')

    inputs = processor(image, user_query, return_tensors="pt")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
def story_vqa_prompt(image_path):
    image_caption = get_image_caption(image_path)
    detected_objects = re.sub('['+string.punctuation+']', '', detect_objects(image_path)).split()
    detected_objects_lst = [detected_object for detected_object in detected_objects if detected_object in words.words()]
    image_text = image_ocr(image_path)
    story_qa_response = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = [{"role":"user","content":f"""You are an image analyzer. You need to come up with questions about different activities happening in an image.
                     The objects detected in the image are {detected_objects_lst}.
                     The caption of the image is {image_caption}.
                     The text present in the image is {image_text}.Use this only it actaully means soething. Igone if it is jibberish.
                    You need tp come up with 5 questions using the above informmation which will help in creating a compelling story about the image.
                     """}],
        temperature = 0.2,
        max_tokens = 2000,
        top_p = 0.95,
        frequency_penalty = 0,
        presence_penalty = 0,
        stop = None
    )
    return story_qa_response['choices'][0]['message']['content']

def image_to_story(image_path):
    image_caption = get_image_caption(image_path)
    image_text = image_ocr(image_path)
    story_qa = story_vqa_prompt(image_path)
    detected_objects = re.sub('['+string.punctuation+']', '', detect_objects(image_path)).split()
    detected_objects_lst = [detected_object for detected_object in detected_objects if detected_object in words.words()]

    image_activity = []
    for detected_object in detected_objects_lst:
        question = f"What is the {detected_object} doing?"
        tmp_activity = visual_qa(image_path,question)
        image_activity.append(tmp_activity)
    image_story1 = openai.ChatCompletion.create(
    model = "gpt-4",
    messages = [{"role":"user","content":f"""You are a world renowned bestseller story teller. You have been given a task to create a story from an image.
                 The image caption is {image_caption}.
                 The objects present in the image are {detected_object}.
                 The test present in the image is {image_text}. Use this only it actaully an English word and means anything. Igone if it is jibberish.
                 These are the different questions you are trying to answer in your story {story_qa}
                 Various activities in the image are {image_activity} 
                 Come up with a compelling and creative story about the image using the above information. Give an exciting title to the story. The title shoudl appear on top of the story.
                    """}],
    temperature = 0.5,
    max_tokens = 2000,
    top_p = 0.95,
    frequency_penalty = 0,
    presence_penalty = 0,
    stop = None
    )
    image_story = image_story1['choices'][0]['message']['content']
    ss["image_story"] = image_story
    image_story = ss["image_story"]
    return image_story


def image_outpainting(image_path,prompt,edit_img_path,size):
    image = Image.open(image_path)
    image = image.resize((256,256),resample=2)
    image.save(edit_img_path+"resized_image.png")

    resized_image = Image.open(edit_img_path+"resized_image.png")
    background = Image.new('RGBA',(1024,1024),(255,255,255,0))
    offset = (300,184)
    background.paste(resized_image, offset)
    bio = BytesIO()
    background.save(bio,format='PNG')
    bio.name = 'test_background.png'
    bio.mode = 'rb'
    bio.seek(0)
    masked_img = BufferedReader(bio)
    image_response = openai.Image.create_edit(
        image=masked_img,
        prompt = "{}".format(prompt),
        n=1,
        size = size,
    )
    url = image_response["data"][0]["url"]
    outpainted_image = requests.get(url)
    with open(edit_img_path+"/outpainted.png", "wb") as f:
        f.write(outpainted_image.content)


def conv_agent(prompt,conversational_memory,user_query,input_str):
    #initialize the agent
    tools = [ImageCaptionTool(), ObjectDetectionTool(),ImageOCRTool(), ImageVQA()]
    print(input_str)
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        prompt=prompt,
        llm=llm,
        max_iterations=5,
        verbose=True,
        memory=conversational_memory,
        early_stopping_method='generate'
    )
    message_response = agent.run(f'{user_query},Here is the action input: {input_str}. Note that entire {input_str} has to be taken as action_input.')
    return message_response
