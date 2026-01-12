import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import datetime
import time
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# --- IMPORTS for Crop Disease
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn

# --- IMPORTS (Voice, SMS, DB) ---
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
from twilio.rest import Client
import pymongo

# --- MODEL CLASSES (Global Scope for Pickling) ---
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

@st.cache_resource
def load_model():
    # Instantiate model structure
    model = ResNet9(3, 38)
    # Load weights
    model.load_state_dict(torch.load('plant_disease_weights.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# -----------------------------------

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agri-Smart BD | এআই মূল্য পূর্বাভাস",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced Professional Dashboard Design
st.markdown("""
    <style>
    /* Main background with gradient */
    .main {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        background-attachment: fixed;
    }
    
    /* Content area styling */
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-top: 1rem;
    }
    
    /* Headers */
    h1 {
        color: #1a1a1a !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* All text elements */
    p, span, div, label, .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Metric styling with gradient backgrounds */
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #1a1a1a !important;
    }
    
    /* Cards effect for metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        border-left: 5px solid #28a745 !important;
    }
    
    .stSuccess > div, .stInfo > div, .stWarning > div {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    .stInfo {
        border-left-color: #17a2b8 !important;
    }
    
    .stWarning {
        border-left-color: #ffc107 !important;
    }
    
    /* Selectbox and input styling */
    .stSelectbox label, .stTextInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Selectbox dropdown styling */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid #11998e !important;
        border-radius: 8px !important;
    }
    
    /* Selectbox selected value */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1a1a1a !important;
    }
    
    /* Dropdown menu options list */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    
    /* Individual dropdown options */
    [role="option"] {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    /* Dropdown option on hover */
    [role="option"]:hover {
        background-color: #11998e !important;
        color: #ffffff !important;
    }
    
    /* Selected option in dropdown */
    [aria-selected="true"] {
        background-color: #38ef7d !important;
        color: #ffffff !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f5132 0%, #198754 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(0,0,0,0.1) !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
    }
    
    /* Footer styling */
    footer {
        color: #1a1a1a !important;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Login Box Styling */
    .login-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATABASE CONNECTION (MONGODB)
# -----------------------------------------------------------------------------
# NOTE: Replace this URI with your actual MongoDB Connection String
# Example: "mongodb+srv://<username>:<password>@cluster0.xyz.mongodb.net/?retryWrites=true&w=majority"
# For Hackathon demo without setup, I will use a local list fallback if connection fails.

MONGO_URI = st.secrets.get("MONGO_URI") or os.getenv("MONGO_URI") or "mongodb+srv://admin:admin123@cluster0.xyz.mongodb.net/?retryWrites=true&w=majority" 

@st.cache_resource
def init_connection():
    try:
        # Connect to MongoDB
        # client = pymongo.MongoClient(MONGO_URI) # Uncomment this when you have real URI
        # return client
        return None # Returning None for demo purpose (In-memory mock)
    except:
        return None

client = init_connection()

# Mock Database for Demo (If MongoDB is not connected)
# Use cache_resource to persist across reruns
@st.cache_resource
def get_mock_db():
    return []

mock_db = get_mock_db()

def get_user(phone):
    """Fetch user from DB"""
    # Real Mongo Implementation:
    # db = client.agri_smart
    # return db.users.find_one({"phone": phone})
    
    # Mock Implementation:
    for user in mock_db:
        if user['phone'] == phone:
            return user
    return None

def create_user(name, phone, district):
    """Insert new user to DB"""
    user_data = {"name": name, "phone": phone, "district": district}
    
    # Real Mongo Implementation:
    # db = client.agri_smart
    # db.users.insert_one(user_data)
    
    # Mock Implementation:
    mock_db.append(user_data)
    return True

# -----------------------------------------------------------------------------
# 3. DATA LOADING FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        price_df = pd.read_csv('bd_crop_price_data.csv')
        prod_df = pd.read_csv('bd_crop_production_data.csv')
        soil_df = pd.read_csv('bd_soil_analysis_data.csv')
        price_df['Price_Date'] = pd.to_datetime(price_df['Price_Date'])
        return price_df, prod_df, soil_df
    except FileNotFoundError:
        return None, None, None

price_df, prod_df, soil_df = load_data()

# --- Weather API Helper ---
@st.cache_data(ttl=3600)
def get_weather_data(city, api_key):
    """Fetch current weather for a city in Bangladesh"""
    if not api_key: return None
    
    # Mapping for OpenWeatherMap (Spelling differences)
    API_CITY_MAPPING = {
        'Cumilla': 'Comilla',
        'Chattogram': 'Chittagong',
        'Barishal': 'Barisal',
        'Jashore': 'Jessore',
        'Bogura': 'Bogra'
    }
    
    search_city = API_CITY_MAPPING.get(city, city)
    
    try:
        # Append ,BD to ensure we get the city in Bangladesh
        url = f"http://api.openweathermap.org/data/2.5/weather?q={search_city},BD&appid={api_key}&units=metric"
        # print(f"weather url-{url}")
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

        return None
    except Exception as e:
        return None

# --- Gemini API Helper ---
def get_gemini_analysis(image, predicted_class, confidence, api_key):
    """
    Get second opinion from Gemini Flash model.
    """
    import time
    
    # Retry configuration
    max_retries = 3
    retry_delay = 2  # seconds

    try:
        genai.configure(api_key=api_key)
        # Using specific model that is generally available
        # If this fails, user might need to check Google AI Studio for enabled models
        model = genai.GenerativeModel('gemini-2.5-flash') 
        
        prompt = f"""
        You are an agricultural expert. I have uploaded an image of a crop leaf.
        My automated ResNet model identified the disease as: '{predicted_class}' with {confidence:.1f}% confidence.
        
        Task:
        1. visually verify if the image likely matches this disease.
        2. Briefly explain the visual symptoms visible in the image.
        3. Suggest organic or chemical remedies suitable for Bangladesh context.
        4. If the image doesn't look like a plant leaf, please state that.
        
        Output in Bengali (Bangla). Keep it concise/bullet points.
        """
        
        # Retry loop for 429 errors
        for attempt in range(max_retries):
            try:
                response = model.generate_content([prompt, image])
                return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1)) # Exponential backoff
                        continue
                    else:
                        return "⚠️ সার্ভার ব্যস্ত আছে (429)। অনুগ্রহ করে একটু পরে আবার চেষ্টা করুন।"
                elif "404" in error_str:
                     # Fallback to older/different model if 1.5-flash fails
                     try:
                        fallback_model = genai.GenerativeModel('gemini-1.0-pro-vision-latest')
                        response = fallback_model.generate_content([prompt, image])
                        return response.text
                     except:
                        return f"মডেল পাওয়া যায়নি (404)। API Key চেক করুন।"
                else:
                    return f"Gemini Analysis Error: {error_str}"
                    
        return "সার্ভার রেসপন্স করছে না।"

    except Exception as e:
        return f"Gemini Setup Error: {str(e)}"

# Dictionaries (Translation)
district_translation = {
    'Dhaka': 'ঢাকা', 'Chittagong': 'চট্টগ্রাম', 'Rajshahi': 'রাজশাহী', 'Khulna': 'খুলনা',
    'Barisal': 'বরিশাল', 'Sylhet': 'সিলেট', 'Rangpur': 'রংপুর', 'Mymensingh': 'ময়মনসিংহ',
    'Comilla': 'কুমিল্লা', 'Gazipur': 'গাজীপুর', 'Narayanganj': 'নারায়ণগঞ্জ', 'Tangail': 'টাঙ্গাইল',
    'Jamalpur': 'জামালপুর', 'Bogra': 'বগুড়া', 'Pabna': 'পাবনা', 'Jessore': 'যশোর',
    'Dinajpur': 'দিনাজপুর', 'Faridpur': 'ফরিদপুর', 'Kushtia': 'কুষ্টিয়া', 'Noakhali': 'নোয়াখালী',
    'Brahmanbaria': 'ব্রাহ্মণবাড়িয়া', 'Feni': 'ফেনী', 'Lakshmipur': 'লক্ষ্মীপুর', 'Chandpur': 'চাঁদপুর',
    'Kishoreganj': 'কিশোরগঞ্জ', 'Netrokona': 'নেত্রকোনা', 'Sherpur': 'শেরপুর', 'Habiganj': 'হবিগঞ্জ',
    'Moulvibazar': 'মৌলভীবাজার', 'Sunamganj': 'সুনামগঞ্জ', 'Narsingdi': 'নরসিংদী', 'Munshiganj': 'মুন্সিগঞ্জ',
    'Manikganj': 'মানিকগঞ্জ', 'Gopalganj': 'গোপালগঞ্জ', 'Madaripur': 'মাদারীপুর', 'Shariatpur': 'শরীয়তপুর',
    'Rajbari': 'রাজবাড়ী', 'Magura': 'মাগুরা', 'Jhenaidah': 'ঝিনাইদহ', 'Narail': 'নড়াইল',
    'Satkhira': 'সাতক্ষীরা', 'Bagerhat': 'বাগেরহাট', 'Pirojpur': 'পিরোজপুর', 'Jhalokati': 'ঝালকাঠি',
    'Patuakhali': 'পটুয়াখালী', 'Barguna': 'বরগুনা', 'Sirajganj': 'সিরাজগঞ্জ', 'Natore': 'নাটোর',
    'Chapainawabganj': 'চাঁপাইনবাবগঞ্জ', 'Naogaon': 'নওগাঁ', 'Joypurhat': 'জয়পুরহাট', 'Gaibandha': 'গাইবান্ধা',
    'Kurigram': 'কুড়িগ্রাম', 'Lalmonirhat': 'লালমনিরহাট', 'Nilphamari': 'নীলফামারী', 'Panchagarh': 'পঞ্চগড়',
    'Thakurgaon': 'ঠাকুরগাঁও', 'Coxs Bazar': 'কক্সবাজার', 'Bandarban': 'বান্দরবান', 'Rangamati': 'রাঙ্গামাটি',
    'Khagrachari': 'খাগড়াছড়ি', 'Meherpur': 'মেহেরপুর', 'Chuadanga': 'চুয়াডাঙ্গা', 'Cumilla': 'কুমিল্লা'
}
crop_translation = {
    'Rice': 'ধান', 'Wheat': 'গম', 'Jute': 'পাট', 'Potato': 'আলু', 'Onion': 'পেঁয়াজ',
    'Garlic': 'রসুন', 'Lentil': 'ডাল', 'Mustard': 'সরিষা', 'Tomato': 'টমেটো',
    'Eggplant': 'বেগুন', 'Cabbage': 'বাঁধাকপি', 'Cauliflower': 'ফুলকপি', 'Chili': 'মরিচ',
    'Cucumber': 'শসা', 'Pumpkin': 'কুমড়া', 'Bitter Gourd': 'করলা', 'Bottle Gourd': 'লাউ',
    'Okra': 'ঢেঁড়স', 'Spinach': 'পালং শাক', 'Coriander': 'ধনিয়া', 'Maize': 'ভুট্টা',
    'Sugarcane': 'আখ', 'Tea': 'চা', 'Mango': 'আম', 'Banana': 'কলা', 'Jackfruit': 'কাঁঠাল',
    'Papaya': 'পেঁপে', 'Guava': 'পেয়ারা', 'Lychee': 'লিচু', 'Pineapple': 'আনারস',
    'Bajra': 'বাজরা', 'Barley': 'যব', 'Chilli': 'মরিচ', 'Citrus': 'লেবুজাতীয় ফল',    
    'Cotton': 'তুলা', 'Cumin': 'জিরা', 'Fennel': 'মৌরি', 'Fenugreek': 'মেথি',
    'Gram': 'ছোলা', 'Oilseeds': 'তেলবীজ', 'Opium': 'আফিম', 'Pomegranate': 'ডালিম', 'Pulses': 'ডালশস্য' 
}
soil_translation = {
    'Clay': 'কর্দম মাটি', 'Loamy': 'দোআঁশ মাটি', 'Sandy': 'বেলে মাটি', 'Silt': 'পলি মাটি',
    'Clay Loam': 'কর্দম দোআঁশ', 'Sandy Loam': 'বেলে দোআঁশ', 'Silty Clay': 'পলি কর্দম',
    'Silty Loam': 'পলি দোআঁশ', 'Peat': 'পিট মাটি', 'Chalky (Calcareous)': 'চুনযুক্ত মাটি',
    'Nitrogenous': 'নাইট্রোজেন সমৃদ্ধ', 'Black lava soil': 'কালো লাভা মাটি'
}

CLASS_LABELS = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
# Bengali translations (optional - expand as needed for your app)
DISEASE_TRANSLATION = {
    'Apple___Apple_scab': 'আপেল স্ক্যাব রোগ',
    'Apple___Black_rot': 'আপেলের কালো পচন রোগ',
    'Apple___Cedar_apple_rust': 'আপেলের সিডার মরিচা রোগ',
    'Apple___healthy': 'আপেল গাছ সুস্থ',

    'Blueberry___healthy': 'ব্লুবেরি গাছ সুস্থ',

    'Cherry_(including_sour)___Powdery_mildew': 'চেরি পাউডারি মিলডিউ রোগ',
    'Cherry_(including_sour)___healthy': 'চেরি গাছ সুস্থ',

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'ভুট্টার সারকোস্পোরা পাতার দাগ রোগ',
    'Corn_(maize)___Common_rust_': 'ভুট্টার সাধারণ মরিচা রোগ',
    'Corn_(maize)___Northern_Leaf_Blight': 'ভুট্টার নর্দান লিফ ব্লাইট রোগ',
    'Corn_(maize)___healthy': 'ভুট্টা গাছ সুস্থ',

    'Grape___Black_rot': 'আঙ্গুরের কালো পচন রোগ',
    'Grape___Esca_(Black_Measles)': 'আঙ্গুরের এসকা (কালো দাগ) রোগ',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'আঙ্গুরের পাতাঝলসানো রোগ',
    'Grape___healthy': 'আঙ্গুর গাছ সুস্থ',

    'Orange___Haunglongbing_(Citrus_greening)': 'কমলার হুয়াংলংবিং (গ্রিনিং) রোগ',

    'Peach___Bacterial_spot': 'পীচ ব্যাকটেরিয়াল দাগ রোগ',
    'Peach___healthy': 'পীচ গাছ সুস্থ',

    'Pepper,_bell___Bacterial_spot': 'ক্যাপসিকাম ব্যাকটেরিয়াল দাগ রোগ',
    'Pepper,_bell___healthy': 'ক্যাপসিকাম গাছ সুস্থ',

    'Potato___Early_blight': 'আলুর আর্লি ব্লাইট রোগ',
    'Potato___Late_blight': 'আলুর লেট ব্লাইট রোগ',
    'Potato___healthy': 'আলু গাছ সুস্থ',

    'Raspberry___healthy': 'রাস্পবেরি গাছ সুস্থ',
    'Soybean___healthy': 'সয়াবিন গাছ সুস্থ',

    'Squash___Powdery_mildew': 'স্কোয়াশ পাউডারি মিলডিউ রোগ',

    'Strawberry___Leaf_scorch': 'স্ট্রবেরির পাতাঝলসানো রোগ',
    'Strawberry___healthy': 'স্ট্রবেরি গাছ সুস্থ',

    'Tomato___Bacterial_spot': 'টমেটো ব্যাকটেরিয়াল দাগ রোগ',
    'Tomato___Early_blight': 'টমেটো আর্লি ব্লাইট রোগ',
    'Tomato___Late_blight': 'টমেটো লেট ব্লাইট রোগ',
    'Tomato___Leaf_Mold': 'টমেটো লিফ মোল্ড রোগ',
    'Tomato___Septoria_leaf_spot': 'টমেটো সেপটোরিয়া পাতার দাগ রোগ',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'টমেটো স্পাইডার মাইট আক্রমণ',
    'Tomato___Target_Spot': 'টমেটো টার্গেট স্পট রোগ',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'টমেটো ইয়েলো লিফ কার্ল ভাইরাস',
    'Tomato___Tomato_mosaic_virus': 'টমেটো মোজাইক ভাইরাস',
    'Tomato___healthy': 'টমেটো গাছ সুস্থ'
}


# Simple remedy suggestions (static dict - expand with real data)
REMEDIES = {
    'Apple___Apple_scab': 'আক্রান্ত পাতা অপসারণ করুন এবং ছত্রাকনাশক স্প্রে করুন।',
    'Apple___Black_rot': 'সংক্রমিত ফল ও ডাল কেটে ফেলুন এবং বাগান পরিষ্কার রাখুন।',
    'Apple___Cedar_apple_rust': 'ছত্রাকনাশক ব্যবহার করুন ও কাছাকাছি জুনিপার গাছ সরান।',
    'Apple___healthy': 'নিয়মিত পরিচর্যা ও সঠিক সার ব্যবহার চালিয়ে যান।',

    'Blueberry___healthy': 'কোন রোগ নেই, নিয়মিত সেচ ও সার প্রয়োগ করুন।',

    'Cherry_(including_sour)___Powdery_mildew': 'সালফার বা উপযুক্ত ছত্রাকনাশক প্রয়োগ করুন।',
    'Cherry_(including_sour)___healthy': 'গাছ সুস্থ, পর্যাপ্ত আলো ও বাতাস নিশ্চিত করুন।',

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'আক্রান্ত পাতা সরিয়ে ফেলুন ও ছত্রাকনাশক দিন।',
    'Corn_(maize)___Common_rust_': 'রোগ সহনশীল জাত ব্যবহার করুন ও প্রয়োজন হলে স্প্রে করুন।',
    'Corn_(maize)___Northern_Leaf_Blight': 'ফসল পর্যায় পরিবর্তন করুন ও ছত্রাকনাশক ব্যবহার করুন।',
    'Corn_(maize)___healthy': 'ভুট্টা গাছ ভালো অবস্থায় আছে।',

    'Grape___Black_rot': 'আক্রান্ত অংশ কেটে ফেলুন ও ছত্রাকনাশক স্প্রে করুন।',
    'Grape___Esca_(Black_Measles)': 'গুরুতর হলে আক্রান্ত গাছ অপসারণ করুন।',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'পাতা পরিষ্কার রাখুন ও ছত্রাকনাশক দিন।',
    'Grape___healthy': 'আঙ্গুর গাছ সুস্থ রয়েছে।',

    'Orange___Haunglongbing_(Citrus_greening)': 'আক্রান্ত গাছ অপসারণ করুন ও পোকা নিয়ন্ত্রণ করুন।',

    'Peach___Bacterial_spot': 'কপার-ভিত্তিক ব্যাকটেরিয়ানাশক ব্যবহার করুন।',
    'Peach___healthy': 'গাছ সুস্থ, পরিচর্যা বজায় রাখুন।',

    'Pepper,_bell___Bacterial_spot': 'আক্রান্ত পাতা সরান ও ব্যাকটেরিয়ানাশক দিন।',
    'Pepper,_bell___healthy': 'ক্যাপসিকাম গাছ ভালো আছে।',

    'Potato___Early_blight': 'ফসল পর্যায় পরিবর্তন ও ছত্রাকনাশক ব্যবহার করুন।',
    'Potato___Late_blight': 'দ্রুত ছত্রাকনাশক স্প্রে করুন ও আক্রান্ত অংশ সরান।',
    'Potato___healthy': 'আলু গাছ সুস্থ রয়েছে।',

    'Raspberry___healthy': 'কোন সমস্যা নেই।',
    'Soybean___healthy': 'সয়াবিন গাছ সুস্থ।',

    'Squash___Powdery_mildew': 'সালফার স্প্রে ও বাতাস চলাচল নিশ্চিত করুন।',

    'Strawberry___Leaf_scorch': 'আক্রান্ত পাতা সরিয়ে ফেলুন।',
    'Strawberry___healthy': 'স্ট্রবেরি গাছ সুস্থ।',

    'Tomato___Bacterial_spot': 'কপার স্প্রে ব্যবহার করুন।',
    'Tomato___Early_blight': 'ছত্রাকনাশক প্রয়োগ করুন।',
    'Tomato___Late_blight': 'আক্রান্ত গাছ দ্রুত অপসারণ করুন।',
    'Tomato___Leaf_Mold': 'গ্রিনহাউসে বাতাস চলাচল বাড়ান।',
    'Tomato___Septoria_leaf_spot': 'পাতা শুকনো রাখুন ও স্প্রে করুন।',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'জৈব কীটনাশক বা পানি স্প্রে করুন।',
    'Tomato___Target_Spot': 'ছত্রাকনাশক প্রয়োগ করুন।',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'সাদা মাছি নিয়ন্ত্রণ করুন।',
    'Tomato___Tomato_mosaic_virus': 'আক্রান্ত গাছ সরিয়ে ফেলুন।',
    'Tomato___healthy': 'টমেটো গাছ সুস্থ রয়েছে।'
}

def translate_bn(text, translation_dict):
    return translation_dict.get(text, text)
def to_bengali_number(number):
    bengali_digits = {'0': '০', '1': '১', '2': '২', '3': '৩', '4': '৪', '5': '৫', '6': '৬', '7': '৭', '8': '৮', '9': '৯', '.': '.'}
    return ''.join(bengali_digits.get(char, char) for char in str(number))

# -----------------------------------------------------------------------------
# 4. AUTHENTICATION LOGIC (TOP RIGHT)
# -----------------------------------------------------------------------------
if 'user' not in st.session_state:
    st.session_state.user = None

# Create a Top Bar Layout
col_logo, col_auth = st.columns([3, 1])

with col_logo:
    st.title("🌾 Agri-Smart BD")

# Auth UI Logic
with col_auth:
    if st.session_state.user:
        # If Logged In
        st.markdown(f"👤 **{st.session_state.user['name']}**")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()
    else:
        # If Not Logged In
        with st.popover("🔐 Login / Sign Up"):
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            
            with tab1:
                st.subheader("লগইন করুন")
                login_phone = st.text_input("মোবাইল নম্বর", key="login_phone")
                if st.button("Login", type="primary"):
                    user = get_user(login_phone)
                    if user:
                        st.session_state.user = user
                        st.success("লগইন সফল!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("নম্বরটি নিবন্ধিত নয়। অনুগ্রহ করে সাইন আপ করুন।")
            
            with tab2:
                st.subheader("নিবন্ধন করুন")
                reg_name = st.text_input("নাম")
                reg_phone = st.text_input("মোবাইল নম্বর", key="reg_phone")
                
                # District List
                district_list = sorted(price_df['District_Name'].unique())
                district_display = {dist: translate_bn(dist, district_translation) for dist in district_list}
                reg_district_bn = st.selectbox("জেলা নির্বাচন করুন", options=list(district_display.values()))
                reg_district = [k for k, v in district_display.items() if v == reg_district_bn][0]
                
                if st.button("Sign Up", type="primary"):
                    if reg_name and reg_phone:
                        existing = get_user(reg_phone)
                        if existing:
                            st.warning("এই নম্বরটি ইতিমধ্যে নিবন্ধিত।")
                        else:
                            create_user(reg_name, reg_phone, reg_district)
                            st.session_state.user = {"name": reg_name, "phone": reg_phone, "district": reg_district}
                            st.success("নিবন্ধন সফল!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.warning("সব তথ্য পূরণ করুন।")

# -----------------------------------------------------------------------------
# 5. MAIN APP CONTENT (Protected or Public)
# -----------------------------------------------------------------------------
# You can choose to hide the whole app if not logged in, or just show it.
# For this request, I will show the app but personalize it if logged in.

if price_df is None:
    st.error("🚨 ডেটাসেট পাওয়া যায়নি!")
    st.stop()

# Helpers
def voice_to_text(audio_bytes):
    r = sr.Recognizer()
    try:
        audio_file = sr.AudioFile(io.BytesIO(audio_bytes))
        with audio_file as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data, language='bn-BD')
        return text
    except:
        return None

def send_sms_alert(to_number, message_body):
    try:
        account_sid = st.secrets.get("TWILIO_ACCOUNT_SID") or os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = st.secrets.get("TWILIO_AUTH_TOKEN") or os.getenv("TWILIO_AUTH_TOKEN")
        from_number = st.secrets.get("TWILIO_PHONE_NUMBER") or os.getenv("TWILIO_PHONE_NUMBER")
        
        if not all([account_sid, auth_token, from_number]):
            return False, "Twilio credentials not configured"
        
        client = Client(account_sid, auth_token)
        message = client.messages.create(body=message_body, from_=from_number, to=to_number)
        return True, message.sid
    except Exception as e:
        return False, str(e)

def get_market_insights(df, current_district, current_crop, days_ahead=7):
    # (Same simplified logic as before)
    insights = {'best_crops_in_district': [], 'best_districts_for_crop': []}
    
    dist_data = df[df['District_Name'] == current_district]
    if not dist_data.empty:
        for crop in dist_data['Crop_Name'].unique():
            crop_df = dist_data[dist_data['Crop_Name'] == crop].sort_values('Price_Date')
            if len(crop_df) > 5:
                try:
                    current_p = crop_df.iloc[-1]['Price_Tk_kg']
                    insights['best_crops_in_district'].append((crop, current_p))
                except: continue
        insights['best_crops_in_district'].sort(key=lambda x: x[1], reverse=True)
        insights['best_crops_in_district'] = insights['best_crops_in_district'][:3]

    crop_data = df[df['Crop_Name'] == current_crop]
    if not crop_data.empty:
        for dist in crop_data['District_Name'].unique():
            dist_df = crop_data[crop_data['District_Name'] == dist].sort_values('Price_Date')
            if len(dist_df) > 5:
                try:
                    current_p = dist_df.iloc[-1]['Price_Tk_kg']
                    insights['best_districts_for_crop'].append((dist, current_p))
                except: continue
        insights['best_districts_for_crop'].sort(key=lambda x: x[1], reverse=True)
        insights['best_districts_for_crop'] = insights['best_districts_for_crop'][:3]
        
    return insights

def get_crop_reasoning(soil_record, crop, yield_val):
    """
    Generate reasoning for why a crop is recommended based on soil conditions
    """
    soil_type = soil_record['Soil_Type']
    ph = soil_record['pH_Level']
    nitrogen = soil_record['Nitrogen_Content_kg_ha']
    organic = soil_record['Organic_Matter_Percent']
    
    reasoning = f"এই অঞ্চলে {crop} চাষের ঐতিহাসিক সাফল্য রয়েছে। "
    
    # pH-based reasoning
    if 6.0 <= ph <= 7.5:
        reasoning += "মাটির পিএইচ স্তর আদর্শ পরিসরে রয়েছে যা এই ফসলের জন্য উপযুক্ত। "
    elif ph < 6.0:
        reasoning += "মাটি কিছুটা অম্লীয় তবে এই ফসল তাতে মানানসই হতে পারে। "
    else:
        reasoning += "মাটি ক্ষারীয় প্রকৃতির, তবে এই ফসল তাতে ভালো জন্মায়। "
    
    # Nitrogen content reasoning
    if nitrogen > 150:
        reasoning += "উচ্চ নাইট্রোজেন সামগ্রী ফসলের বৃদ্ধিতে সহায়ক। "
    elif nitrogen > 100:
        reasoning += "মাঝারি নাইট্রোজেন স্তর পর্যাপ্ত। "
    else:
        reasoning += "নাইট্রোজেন সার প্রয়োগ বিবেচনা করুন। "
    
    # Organic matter reasoning
    if organic > 2.0:
        reasoning += f"উচ্চ জৈব পদার্থ ({organic:.1f}%) মাটির উর্বরতা নিশ্চিত করে। "
    
    # Yield-based reasoning
    reasoning += f"ঐতিহাসিক তথ্য অনুযায়ী, গড় ফলন {yield_val:.1f} কুইন্টাল/হেক্টর অর্জন করা সম্ভব।"
    
    return reasoning

# --- Sidebar ---
st.sidebar.markdown("**এআই চালিত কৃষি বুদ্ধিমত্তা**")
menu = st.sidebar.radio("মডিউল নির্বাচন করুন:", ["📊 মূল্য পূর্বাভাস (এআই)", "💰 সেরা বাজার খুঁজুন", "🌱 মাটি ও ফসল পরামর্শদাতা", "🦠 ফসল বিষাক্তি পরিচিতি"])

# -----------------------------------------------------------------------------
# MODULE 1: AI PRICE FORECASTING
# -----------------------------------------------------------------------------
if menu == "📊 মূল্য পূর্বাভাস (এআই)":
    st.markdown("### মেশিন লার্নিং ব্যবহার করে ৩০ দিনের আগাম মূল্যের পূর্বাভাস।")
    
    # Auto-select district if logged in
    district_list = sorted(price_df['District_Name'].unique())
    district_display = {dist: translate_bn(dist, district_translation) for dist in district_list}
    district_options_list = list(district_display.values())
    
    # Session State Logic for District
    if 'selected_district_val' not in st.session_state:
        # Default to User's District if logged in
        if st.session_state.user:
            user_dist_bn = translate_bn(st.session_state.user['district'], district_translation)
            if user_dist_bn in district_options_list:
                st.session_state.selected_district_val = user_dist_bn
            else:
                st.session_state.selected_district_val = district_options_list[0]
        else:
            st.session_state.selected_district_val = district_options_list[0]

    # Voice Input
    c1, c2 = st.columns([1, 4])
    with c1:
        audio = mic_recorder(start_prompt="🎤 বলুন", stop_prompt="🛑 থামুন", key='recorder', format="wav", use_container_width=True)
    
    voice_text = ""
    if audio:
        with st.spinner("প্রসেস হচ্ছে..."):
            voice_text = voice_to_text(audio['bytes'])
        if voice_text:
            st.success(f"🗣️ আপনি বলেছেন: **'{voice_text}'**")
            for dist_bn in district_options_list:
                    st.toast(f"✅ জেলা শনাক্ত হয়েছে: {dist_bn}")
                    break
    
    # Geolocation Button
    if st.button("📍 আমার বর্তমান অবস্থান ব্যবহার করুন"):
        try:
             # Using ipinfo.io for IP-based location (Free tier, no key needed usually)
            loc_response = requests.get("https://ipinfo.io/json", timeout=10)
            if loc_response.status_code == 200:
                loc_data = loc_response.json()
                city = loc_data.get('city', '')
                
                # Try to fuzzy match or direct match with available districts
                matched_district = None
                
                # 1. Direct Match
                if city in district_display:
                     matched_district = city
                
                # 2. Check Alias/Mapping if needed
                if not matched_district:
                     # Reverse mapping of API_CITY_MAPPING could be useful here
                     # or check if city is in values of district_translation (English)
                     pass

                if matched_district:
                     bn_dist = district_display.get(matched_district)
                     if bn_dist in district_options_list:
                         st.session_state.selected_district_val = bn_dist
                         st.success(f"📍 আপনার অবস্থান শনাক্ত হয়েছে: {city}")
                         st.rerun()
                else:
                    st.warning(f"⚠️ আপনার শহর ({city}) আমাদের ডেটাসেটে পাওয়া যায়নি। অনুগ্রহ করে তালিকা থেকে নির্বাচন করুন।")
            else:
                st.error("অবস্থান নির্ণয় করা সম্ভব হয়নি।")
        except Exception as e:
            st.error(f"অবস্থান এরর: {str(e)}")

    st.divider()

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        selected_district_bn = st.selectbox("📍 জেলা নির্বাচন করুন", options=district_options_list, key='selected_district_val')
        selected_district = [k for k, v in district_display.items() if v == selected_district_bn][0]
    
    with col2:
        available_crops = sorted(price_df[price_df['District_Name'] == selected_district]['Crop_Name'].unique())
        crop_display = {crop: translate_bn(crop, crop_translation) for crop in available_crops}
        crop_options_list = list(crop_display.values())
        
        crop_index = 0
        if voice_text:
            for i, crop_bn in enumerate(crop_options_list):
                if crop_bn in voice_text:
                    crop_index = i
                    break
        
        selected_crop_bn = st.selectbox("🌽 ফসল নির্বাচন করুন", options=crop_options_list, index=crop_index, format_func=lambda x: x)
        selected_crop = [k for k, v in crop_display.items() if v == selected_crop_bn][0]

    # --- WEATHER INTEGRATION ---
    weather_icon_url = None
    weather_advice = ""
    
    # Try to get API Key from secrets, env, or input
    weather_api_key = st.secrets.get("WEATHER_API_KEY") or os.getenv("WEATHER_API_KEY")
    if not weather_api_key:
        with st.expander("☁️ আবহাওয়া সেটিংস (API Key)"):
            weather_api_key = st.text_input("OpenWeatherMap API Key দিন:", type="password", key="w_key")
    
    if weather_api_key:
        w_data = get_weather_data(selected_district, weather_api_key)
        if w_data:
            main = w_data['main']
            weather_desc = w_data['weather'][0]['description']
            icon_code = w_data['weather'][0]['icon']
            weather_icon_url = f"http://openweathermap.org/img/wn/{icon_code}@2x.png"
            
            # Simple Advisory Logic
            if 'rain' in weather_desc.lower() or 'drizzle' in weather_desc.lower() or 'thunderstorm' in weather_desc.lower():
                weather_advice = "🌧️ **সতর্কতা:** বৃষ্টির সম্ভাবনা। ফসল সংগ্রহ বা সার প্রয়োগ থেকে বিরত থাকুন।"
            elif main['temp'] > 35:
                weather_advice = "☀️ **সতর্কতা:** অতিরিক্ত তাপমাত্রা। জমিতে পর্যাপ্ত সেচ নিশ্চিত করুন।"
            elif main['humidity'] > 85:
                 weather_advice = "💧 **সতর্কতা:** উচ্চ আর্দ্রতা। ছত্রাকজনিত রোগের ঝুঁকি বেশি।"
            else:
                 weather_advice = "✅ আবহাওয়া চাষাবাদের অনুকূল।"

            # Display Weather Compactly
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                <div style="display: flex; align-items: center;">
                    <img src="{weather_icon_url}" width="60">
                    <div style="margin-left: 10px;">
                        <h4 style="margin: 0; color: #1565c0;">{translate_bn(selected_district, district_translation)} আবহাওয়া</h4>
                        <p style="margin: 0; font-size: 16px;"><b>{main['temp']}°C</b> | {weather_desc.title()} | আর্দ্রতা: {main['humidity']}%</p>
                    </div>
                </div>
                <div style="background-color: #fff; padding: 10px; border-radius: 8px; border-left: 4px solid #ff9800;">
                    {weather_advice}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Analysis & Prediction
    filtered_df = price_df[(price_df['District_Name'] == selected_district) & (price_df['Crop_Name'] == selected_crop)].sort_values('Price_Date')

    if len(filtered_df) > 10:
        # Feature Engineering
        filtered_df['Date_Ordinal'] = filtered_df['Price_Date'].map(datetime.datetime.toordinal)
        filtered_df['Month'] = filtered_df['Price_Date'].dt.month
        filtered_df['Week'] = filtered_df['Price_Date'].dt.isocalendar().week
        filtered_df['Year'] = filtered_df['Price_Date'].dt.year
        
        X = filtered_df[['Date_Ordinal', 'Month', 'Week', 'Year']]
        y = filtered_df['Price_Tk_kg']
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        last_date = filtered_df['Price_Date'].max()
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]
        future_data = pd.DataFrame({'Price_Date': future_dates})
        future_data['Date_Ordinal'] = future_data['Price_Date'].map(datetime.datetime.toordinal)
        future_data['Month'] = future_data['Price_Date'].dt.month
        future_data['Week'] = future_data['Price_Date'].dt.isocalendar().week
        future_data['Year'] = future_data['Price_Date'].dt.year
        
        # Get predictions with confidence intervals
        predictions = model.predict(future_data[['Date_Ordinal', 'Month', 'Week', 'Year']])
        
        # Calculate confidence intervals using tree predictions
        tree_predictions = np.array([tree.predict(future_data[['Date_Ordinal', 'Month', 'Week', 'Year']]) for tree in model.estimators_])
        std_predictions = tree_predictions.std(axis=0)
        
        future_data['Predicted_Price'] = predictions
        future_data['Upper_Bound'] = predictions + 1.96 * std_predictions
        future_data['Lower_Bound'] = predictions - 1.96 * std_predictions
        
        # Plot with confidence intervals
        st.subheader(f"মূল্য প্রবণতা: {translate_bn(selected_crop, crop_translation)}")
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=filtered_df['Price_Date'], 
            y=filtered_df['Price_Tk_kg'], 
            mode='lines', 
            name='ঐতিহাসিক', 
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Predicted data
        fig.add_trace(go.Scatter(
            x=future_data['Price_Date'], 
            y=future_data['Predicted_Price'], 
            mode='lines', 
            name='পূর্বাভাস', 
            line=dict(color='#00cc96', width=2)
        ))
        
        # Confidence interval upper bound
        fig.add_trace(go.Scatter(
            x=future_data['Price_Date'],
            y=future_data['Upper_Bound'],
            mode='lines',
            name='উর্ধ্ব সীমা',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Confidence interval lower bound with fill
        fig.add_trace(go.Scatter(
            x=future_data['Price_Date'],
            y=future_data['Lower_Bound'],
            mode='lines',
            name='নিম্ন সীমা',
            line=dict(width=0),
            fillcolor='rgba(0, 204, 150, 0.2)',
            fill='tonexty',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            xaxis_title='তারিখ',
            yaxis_title='মূল্য (৳/কেজি)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

        current_price = filtered_df.iloc[-1]['Price_Tk_kg']
        avg_price = predictions.mean()
        trend = "উর্ধ্বমুখী 📈" if avg_price > current_price else "নিম্নমুখী 📉"
        
        m1, m2, m3 = st.columns(3)
        m1.metric("বর্তমান মূল্য", f"৳ {to_bengali_number(f'{current_price:.2f}')}")
        m2.metric("গড় পূর্বাভাস", f"৳ {to_bengali_number(f'{avg_price:.2f}')}")
        m3.metric("প্রবণতা", trend)

        # SMS Alert Section (Personalized)
        st.markdown("---")
        st.subheader("📲 স্মার্ট এসএমএস অ্যালার্ট")
        
        c_sms1, c_sms2 = st.columns([2, 1])
        with c_sms1:
            # Autofill phone number if logged in
            default_phone = st.session_state.user['phone'] if st.session_state.user else "+18777804236"
            phone_number = st.text_input("মোবাইল নম্বর", value=default_phone)
        
        with c_sms2:
            st.write("")
            st.write("")
            send_btn = st.button("🚀 পাঠান", type="primary", use_container_width=True)
            
        if send_btn:
            # Login check enforcement (Optional, but adds value)
            if not st.session_state.user:
                st.warning("⚠️ অনুগ্রহ করে এসএমএস পেতে লগইন করুন।")
            else:
                with st.spinner("অ্যালার্ট জেনারেট হচ্ছে..."):
                    insights = get_market_insights(price_df, selected_district, selected_crop)
                    
                    msg = f"সতর্কতা: {selected_district_bn}তে {selected_crop_bn} ৳{int(current_price)}।"
                    if insights['best_districts_for_crop']:
                        top_dist, top_price = insights['best_districts_for_crop'][0]
                        if top_price > current_price:
                            d_bn = translate_bn(top_dist, district_translation)
                            msg += f" বেশি দাম: {d_bn}তে ৳{int(top_price)}।"
                        else:
                            msg += " এখানের দামই সেরা।"
                    msg += " -AgriSmart"
                    msg = msg[:158]
                    
                    success, response = send_sms_alert(phone_number, msg)
                    if success:
                        st.success("✅ এসএমএস পাঠানো হয়েছে!")
                        st.balloons()
                    else:
                        st.error(f"❌ ব্যর্থ: {response}")

# -----------------------------------------------------------------------------
# MODULE 2: BEST MARKET FINDER
# -----------------------------------------------------------------------------
elif menu == "💰 সেরা বাজার খুঁজুন":
    st.title("💰 সেরা বাজার খুঁজুন")
    st.divider()

    all_crops = sorted(price_df['Crop_Name'].unique())
    all_crops_display = {crop: translate_bn(crop, crop_translation) for crop in all_crops}
    target_crop_bn = st.selectbox("🔍 ফসল নির্বাচন করুন", options=list(all_crops_display.values()))
    target_crop = [k for k, v in all_crops_display.items() if v == target_crop_bn][0]

    transport_cost = st.number_input("পরিবহন খরচ (টাকা/কেজি)", min_value=0.0, value=2.0)

    latest_date = price_df['Price_Date'].max()
    recent_data = price_df[(price_df['Crop_Name'] == target_crop) & (price_df['Price_Date'] >= latest_date - datetime.timedelta(days=60))]
    market_data = recent_data.sort_values('Price_Date').groupby('District_Name').tail(1).copy()

    if not market_data.empty:
        market_data['Net_Profit'] = market_data['Price_Tk_kg'] - transport_cost
        best_market = market_data.sort_values('Net_Profit', ascending=False).iloc[0]
        
        # Enhanced Net Profit Visualization with highlighted card
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 2rem; 
                    border-radius: 15px; 
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                    text-align: center;
                    margin: 1rem 0;'>
            <h2 style='color: white; margin: 0; font-size: 1.5rem;'>🏆 সেরা বাজার</h2>
            <h1 style='color: #ffffff; margin: 0.5rem 0; font-size: 2.5rem;'>{translate_bn(best_market['District_Name'], district_translation)}</h1>
            <h3 style='color: white; margin: 0;'>নিট লাভ: ৳{to_bengali_number(f"{best_market['Net_Profit']:.2f}")}/কেজি</h3>
            <p style='color: rgba(255,255,255,0.9); margin-top: 1rem;'>মূল্য: ৳{to_bengali_number(f"{best_market['Price_Tk_kg']:.2f}")} | পরিবহন: ৳{to_bengali_number(f"{transport_cost:.2f}")}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📊 সকল জেলার তুলনা")
        fig = px.bar(
            market_data.sort_values('Net_Profit', ascending=True), 
            x='Net_Profit', 
            y='District_Name', 
            orientation='h', 
            color='Net_Profit', 
            color_continuous_scale='Greens',
            labels={'Net_Profit': 'নিট লাভ (৳/কেজি)', 'District_Name': 'জেলা'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# MODULE 3: SOIL ADVISOR
# -----------------------------------------------------------------------------
elif menu == "🌱 মাটি ও ফসল পরামর্শদাতা":
    st.title("🌱 ফসল পরামর্শদাতা")
    st.divider()

    soil_districts = sorted(soil_df['District_Name'].unique())
    soil_district_display = {dist: translate_bn(dist, district_translation) for dist in soil_districts}
    
    # Auto-select if logged in
    default_idx = 0
    if st.session_state.user:
        u_dist = translate_bn(st.session_state.user['district'], district_translation)
        vals = list(soil_district_display.values())
        if u_dist in vals:
            default_idx = vals.index(u_dist)

    target_district_bn = st.selectbox("📍 অবস্থান নির্বাচন করুন", options=list(soil_district_display.values()), index=default_idx)
    target_district = [k for k, v in soil_district_display.items() if v == target_district_bn][0]

    soil_record = soil_df[soil_df['District_Name'] == target_district].iloc[0]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("মাটি", translate_bn(soil_record['Soil_Type'], soil_translation))
    c2.metric("পিএইচ", to_bengali_number(f"{soil_record['pH_Level']:.2f}"))
    c3.metric("নাইট্রোজেন", f"{to_bengali_number(f'{soil_record['Nitrogen_Content_kg_ha']:.1f}')}")
    c4.metric("জৈব", f"{to_bengali_number(f'{soil_record['Organic_Matter_Percent']:.1f}')}%")

    st.subheader("🌾 সুপারিশকৃত ফসল")
    dist_prod = prod_df[prod_df['District_Name'] == target_district]
    top_crops = dist_prod.groupby('Crop_Name')['Yield_Quintals_per_Ha'].mean().sort_values(ascending=False).head(5)

    # Enhanced crop recommendations with reasoning
    for idx, (crop, yield_val) in enumerate(top_crops.items(), 1):
        # Get reasoning based on soil conditions
        reasoning = get_crop_reasoning(soil_record, crop, yield_val)
        
        with st.expander(f"#{idx} {translate_bn(crop, crop_translation)} - ঐতিহাসিক ফলন: {to_bengali_number(f'{yield_val:.1f}')} কুইন্টাল/হেক্টর"):
            st.markdown(f"**কেন এই ফসলটি উপযুক্ত:**")
            st.write(reasoning)
elif menu == "🦠 ফসল বিষাক্তি পরিচিতি":
    st.title("🦠 ফসল বিষাক্তি পরিচিতি")
    st.markdown("Upload a photo of your crop leaf for AI analysis (99.2% accuracy on global dataset). Note: This is for guidance only—consult local agri experts for confirmation.")


    model = load_model()
    
    # UI Layout: Tabs for Input Method
    tab_cam, tab_up = st.tabs(["📸 ছবি তুলুন", "📂 ছবি আপলোড করুন"])
    
    img_file = None
    
    with tab_cam:
        cam_img = st.camera_input("ফসল বা পাতার ছবি তুলুন")
        if cam_img:
            img_file = cam_img
            
    with tab_up:
        up_img = st.file_uploader("গ্যালারি থেকে ছবি নির্বাচন করুন (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if up_img:
            img_file = up_img

    if img_file:
        # Display Image
        image = Image.open(img_file)
        
        # Center the image
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(image, caption="বিশ্লেষণকৃত ছবি", use_container_width=True)

        with st.spinner("রোগ নির্ণয় করা হচ্ছে..."):
            # Transformation
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ])
            img_tensor = transform(image).unsqueeze(0)

            # Inference
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                pred_class = CLASS_LABELS[predicted.item()]
                conf_score = confidence.item() * 100

        # Display Results
        bn_label = DISEASE_TRANSLATION.get(pred_class, pred_class)
        remedy = REMEDIES.get(pred_class, "পরামর্শ: স্থানীয় কৃষি কর্মকর্তার সাথে যোগাযোগ করুন।")
        
        st.divider()
        st.subheader("ফলাফল:")
        
        # Result Badge
        if "healthy" in pred_class:
            st.success(f"✅ **অবস্থা:** {bn_label}")
        else:
            st.error(f"⚠️ **শনাক্ত রোগ:** {bn_label}")
            
        # Confidence Bar
        st.write(f"**সঠিকতার হার:** {conf_score:.1f}%")
        st.progress(int(conf_score))
        
        # --- GEMINI INTEGRATION ---
        gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if gemini_api_key:
            with st.expander("🤖 এআই বিশেষজ্ঞের মতামত (Gemini 2.0)", expanded=True):
                with st.spinner("Gemini চিত্র বিশ্লেষণ করছে..."):
                    gemini_response = get_gemini_analysis(image, pred_class, conf_score, gemini_api_key)
                    st.markdown(gemini_response)
        
        # Remedy Section
        with st.container():
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
                <h4 style="color: #31333F;">💡 পরামর্শ ও প্রতিকার</h4>
                <p style="font-size: 16px;">{remedy}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Disclaimer
        with st.expander("⚠️ দাবিত্যাগ (Disclaimer)"):
            st.write("এই এআই মডেলটি সহায়ক টুল হিসেবে তৈরি। এটি ৯৯.২% নির্ভুল হলেও, চূড়ান্ত সিদ্ধান্তের জন্য সর্বদা কৃষি বিশেষজ্ঞের পরামর্শ নিন।")

# Footer
st.markdown("<br><hr><div style='text-align: center; color: #555;'>Agri-Smart BD | Built for AI Build-a-thon 2025</div>", unsafe_allow_html=True)