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
import warnings
# Suppress Google Generative AI and other FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
warnings.filterwarnings("ignore", category=FutureWarning, module="keras.src.export.tf2onnx_lib")

import google.generativeai as genai

# Load environment variables
# Load environment variables
load_dotenv()

# Set Hugging Face Hub to Offline Mode for Demo
os.environ["HF_HUB_OFFLINE"] = "1"
# Suppress TensorFlow Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# --- IMPORTS for Crop Disease (Linear/Hugging Face)
import tensorflow as tf
import tf_keras # Use legacy Keras for .h5 model compatibility
from tf_keras.models import load_model as load_keras_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image, ImageOps
import numpy as np
import cv2 # For leaf detection filter

# --- IMPORTS (Voice, SMS, DB) ---
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
from twilio.rest import Client
import pymongo
from streamlit_js_eval import streamlit_js_eval
from streamlit_option_menu import option_menu

# --- MODEL LOADING (Hugging Face - Offline) ---
@st.cache_resource
def load_plant_model():
    model_path = "./plant_disease_model/plant_disease_efficientnetb4.h5"
    try:
        # Load from local .h5 file directly (Keras format)
        model = load_keras_model(model_path)
        return model
    except Exception as e:
        st.error(f"Critical Error Loading Model: {e}")
        return None

def is_likely_leaf(img_pil):
    """
    Advanced heuristic to check if image is a real leaf.
    Checks:
    1. Green Content (Color)
    2. Texture/Pattern (Edges) - Rejects smooth green screens
    """
    try:
        # Convert PIL to OpenCV format (RGB -> BGR)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # 1. Color Check (HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        green_pixels = np.count_nonzero(mask)
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = green_pixels / total_pixels
        
        # 2. Texture Check (Canny Edge Detection)
        # Real leaves have veins/texture; Green screens are smooth.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur slightly to remove noise
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray_blurred, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        edge_ratio = edge_pixels / total_pixels
        
        # LOGIC:
        # - Must be at least 15% green
        # - Must have some texture (> 0.5% edges) to reject flat screens
        
        is_green = green_ratio > 0.15
        has_texture = edge_ratio > 0.005 # 0.5% edge density
        
        if not is_green:
             return False, f"⚠️ এটা পাতার ছবি মনে হচ্ছে না (সবুজের পরিমাণ: {green_ratio:.1%})। দয়া করে ফসলের বা পাতার স্পষ্ট ছবি দিন।"

        # If it's green but super smooth (low edges), it's likely artificial
        if is_green and not has_texture:
            return False, f"⚠️ কৃত্রিম বা স্ক্রিন মনে হচ্ছে (সবুজের পরিমাণ: {green_ratio:.1%}, টেক্সচার: {edge_ratio:.1%})। আসল পাতার ছবি দিন।"
            
        return True, "Analysis Proceeding"
    except Exception:
        return True, "Error bypassed" # Fail safe

# -----------------------------------

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agri-Smart BD | এআই মূল্য পূর্বাভাস",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HIDE STREAMLIT STYLE (PWA Fix) ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stApp > header {display: none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

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
    
    /* --- RESPONSIVE MENU CSS --- */
    
    /* Mobile View (< 768px): Hide Sidebar, Show Top Menu */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {
            display: none;
        }
    }

    /* Desktop View (>= 768px): Hide Top Menu, Show Sidebar */
    @media (min-width: 768px) {
        /* Use the adjacent sibling selector to look for the specific marker we will inject */
        div:has(#mobile-menu-marker) + [data-testid="stExpander"] {
            display: none;
        }
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

@st.cache_data(ttl=3600)
def get_weather_forecast(city, api_key):
    """Fetch 5-day weather forecast"""
    if not api_key: return None
    
    API_CITY_MAPPING = {
        'Cumilla': 'Comilla', 'Chattogram': 'Chittagong', 'Barishal': 'Barisal',
        'Jashore': 'Jessore', 'Bogura': 'Bogra'
    }
    search_city = API_CITY_MAPPING.get(city, city)
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={search_city},BD&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

# --- WEATHER & FORECAST HELPERS (COORDINATES BASED) ---
@st.cache_data(ttl=3600)
def get_weather_by_coords(lat, lon, api_key):
    """Fetch current weather using Pin-Point Coordinates"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=3600)
def get_forecast_by_coords(lat, lon, api_key):
    """Fetch 5-day forecast to check rain probability for tomorrow"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
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
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 
    'Soybean___healthy', 
    'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
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


# AI Doctor Prescription Map (Actionable Advice)
CROP_PRESCRIPTION_MAP = {

    # ================= APPLE =================
    'Apple___Apple_scab': {
        "cause": "Venturia inaequalis ছত্রাক",
        "solution": "আক্রান্ত পাতা ও ফল অপসারণ করুন। বাগান পরিষ্কার রাখুন।",
        "medicine": "Score 250 EC / Dithane M-45",
        "dosage": "প্রতি লিটার পানিতে ০.৫ মিলি Score অথবা ২ গ্রাম Dithane মিশিয়ে স্প্রে করুন।"
    },
    'Apple___Black_rot': {
        "cause": "Botryosphaeria ছত্রাক",
        "solution": "সংক্রমিত ডাল ও ফল কেটে পুড়িয়ে ফেলুন।",
        "medicine": "Copper Fungicide",
        "dosage": "২ গ্রাম প্রতি লিটার পানিতে মিশিয়ে স্প্রে করুন।"
    },
    'Apple___Cedar_apple_rust': {
        "cause": "Gymnosporangium ছত্রাক",
        "solution": "কাছাকাছি জুনিপার গাছ সরান।",
        "medicine": "Bayleton 25 WP",
        "dosage": "১ গ্রাম প্রতি লিটার পানিতে মিশিয়ে স্প্রে করুন।"
    },
    'Apple___healthy': {
        "cause": "কোন রোগ নেই",
        "solution": "নিয়মিত সার ও সেচ বজায় রাখুন।",
        "medicine": "প্রযোজ্য নয়",
        "dosage": "-"
    },

    # ================= BLUEBERRY =================
    'Blueberry___healthy': {
        "cause": "কোন রোগ নেই",
        "solution": "সঠিক pH ও সেচ বজায় রাখুন।",
        "medicine": "প্রযোজ্য নয়",
        "dosage": "-"
    },

    # ================= CHERRY =================
    'Cherry_(including_sour)___Powdery_mildew': {
        "cause": "Podosphaera ছত্রাক",
        "solution": "আলো ও বাতাস চলাচল নিশ্চিত করুন।",
        "medicine": "Sulphur Fungicide",
        "dosage": "২ গ্রাম প্রতি লিটার পানিতে মিশিয়ে স্প্রে করুন।"
    },
    'Cherry_(including_sour)___healthy': {
        "cause": "কোন রোগ নেই",
        "solution": "গাছ সুস্থ আছে।",
        "medicine": "প্রযোজ্য নয়",
        "dosage": "-"
    },

    # ================= CORN =================
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        "cause": "Cercospora ছত্রাক",
        "solution": "ফসল পর্যায়ক্রম অনুসরণ করুন।",
        "medicine": "Tilt 250 EC",
        "dosage": "০.৫ মিলি প্রতি লিটার পানিতে স্প্রে করুন।"
    },
    'Corn_(maize)___Common_rust_': {
        "cause": "Puccinia sorghi ছত্রাক",
        "solution": "রোগ সহনশীল জাত ব্যবহার করুন।",
        "medicine": "Score 250 EC",
        "dosage": "০.৫ মিলি প্রতি লিটার পানিতে স্প্রে করুন।"
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        "cause": "Exserohilum turcicum ছত্রাক",
        "solution": "পরিষ্কার বীজ ব্যবহার করুন।",
        "medicine": "Tilt 250 EC",
        "dosage": "০.৫ মিলি প্রতি লিটার পানিতে স্প্রে করুন।"
    },
    'Corn_(maize)___healthy': {
        "cause": "কোন রোগ নেই",
        "solution": "ভুট্টা গাছ সুস্থ।",
        "medicine": "প্রযোজ্য নয়",
        "dosage": "-"
    },

    # ================= GRAPE =================
    'Grape___Black_rot': {
        "cause": "Guignardia bidwellii ছত্রাক",
        "solution": "আক্রান্ত অংশ কেটে ফেলুন।",
        "medicine": "Dithane M-45",
        "dosage": "২ গ্রাম প্রতি লিটার পানিতে স্প্রে করুন।"
    },
    'Grape___Esca_(Black_Measles)': {
        "cause": "ছত্রাকজনিত জটিল রোগ",
        "solution": "গুরুতর হলে গাছ অপসারণ করুন।",
        "medicine": "কার্যকর চিকিৎসা নেই",
        "dosage": "-"
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        "cause": "Isariopsis ছত্রাক",
        "solution": "পাতা পরিষ্কার রাখুন।",
        "medicine": "Copper Fungicide",
        "dosage": "২ গ্রাম প্রতি লিটার পানিতে স্প্রে করুন।"
    },
    'Grape___healthy': {
        "cause": "কোন রোগ নেই",
        "solution": "গাছ সুস্থ।",
        "medicine": "প্রযোজ্য নয়",
        "dosage": "-"
    },

    # ================= ORANGE =================
    'Orange___Haunglongbing_(Citrus_greening)': {
        "cause": "ব্যাকটেরিয়া (Candidatus Liberibacter)",
        "solution": "আক্রান্ত গাছ অপসারণ করুন।",
        "medicine": "Imidacloprid",
        "dosage": "০.৫ মিলি প্রতি লিটার পানিতে স্প্রে করুন।"
    },

    # ================= PEACH =================
    'Peach___Bacterial_spot': {
        "cause": "Xanthomonas ব্যাকটেরিয়া",
        "solution": "বৃষ্টিতে স্প্রে এড়িয়ে চলুন।",
        "medicine": "Copper Oxychloride",
        "dosage": "২ গ্রাম প্রতি লিটার পানিতে স্প্রে করুন।"
    },
    'Peach___healthy': {
        "cause": "কোন রোগ নেই",
        "solution": "গাছ সুস্থ।",
        "medicine": "প্রযোজ্য নয়",
        "dosage": "-"
    },

    # ================= PEPPER =================
    'Pepper,_bell___Bacterial_spot': {
        "cause": "Xanthomonas ব্যাকটেরিয়া",
        "solution": "পাতায় পানি জমতে দেবেন না।",
        "medicine": "Kocide 3000",
        "dosage": "২.৫ গ্রাম প্রতি লিটার পানিতে স্প্রে করুন।"
    },
    'Pepper,_bell___healthy': {
        "cause": "কোন রোগ নেই",
        "solution": "গাছ ভালো আছে।",
        "medicine": "প্রযোজ্য নয়",
        "dosage": "-"
    },

    # ================= RASPBERRY =================
    'Raspberry___healthy': {
        "cause": "কোন রোগ নেই",
        "solution": "নিয়মিত পরিচর্যা চালিয়ে যান।",
        "medicine": "প্রযোজ্য নয়",
        "dosage": "-"
    },

    # ================= SOYBEAN =================
    'Soybean___healthy': {
        "cause": "কোন রোগ নেই",
        "solution": "সয়াবিন গাছ সুস্থ।",
        "medicine": "প্রযোজ্য নয়",
        "dosage": "-"
    },

    # ================= SQUASH =================
    'Squash___Powdery_mildew': {
        "cause": "Erysiphe ছত্রাক",
        "solution": "বাতাস চলাচল নিশ্চিত করুন।",
        "medicine": "Sulphur",
        "dosage": "২ গ্রাম প্রতি লিটার পানিতে স্প্রে করুন।"
    },

    # ================= STRAWBERRY =================
    'Strawberry___Leaf_scorch': {
        "cause": "Diplocarpon ছত্রাক",
        "solution": "আক্রান্ত পাতা সরিয়ে ফেলুন।",
        "medicine": "Dithane M-45",
        "dosage": "২ গ্রাম প্রতি লিটার পানিতে স্প্রে করুন।"
    },
    'Strawberry___healthy': {
        "cause": "কোন রোগ নেই",
        "solution": "গাছ সুস্থ।",
        "medicine": "প্রযোজ্য নয়",
        "dosage": "-"
    },

    # ================= TOMATO (remaining) =================
    'Tomato___Septoria_leaf_spot': {
        "cause": "Septoria lycopersici ছত্রাক",
        "solution": "পাতা শুকনো রাখুন।",
        "medicine": "Score 250 EC",
        "dosage": "০.৫ মিলি প্রতি লিটার পানিতে স্প্রে করুন।"
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        "cause": "মাকড় জাতীয় পোকা",
        "solution": "পাতার নিচে পানি স্প্রে করুন।",
        "medicine": "Vertimec",
        "dosage": "০.৫ মিলি প্রতি লিটার পানিতে স্প্রে করুন।"
    },
    'Tomato___Target_Spot': {
        "cause": "Corynespora ছত্রাক",
        "solution": "পরিষ্কার চাষাবাদ বজায় রাখুন।",
        "medicine": "Nativo 75 WG",
        "dosage": "০.৬ গ্রাম প্রতি লিটার পানিতে স্প্রে করুন।"
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        "cause": "ভাইরাস (Whitefly বাহক)",
        "solution": "সাদা মাছি নিয়ন্ত্রণ করুন।",
        "medicine": "Imidacloprid",
        "dosage": "০.৫ মিলি প্রতি লিটার পানিতে স্প্রে করুন।"
    },
    'Tomato___Tomato_mosaic_virus': {
        "cause": "ভাইরাস",
        "solution": "আক্রান্ত গাছ অপসারণ করুন।",
        "medicine": "কার্যকর ওষুধ নেই",
        "dosage": "-"
    },
    'Tomato___healthy': {
        "cause": "কোন রোগ নেই",
        "solution": "টমেটো গাছ সুস্থ।",
        "medicine": "প্রযোজ্য নয়",
        "dosage": "-"
    },

    # ================= POTATO =================
    'Potato___Early_blight': {
        "cause": "Alternaria solani ছত্রাক",
        "solution": "আক্রান্ত পাতা সংগ্রহ করে নষ্ট করুন। ফসল পর্যায়ক্রম অনুসরণ করুন।",
        "medicine": "Dithane M-45 / Amistar Top",
        "dosage": "প্রতি লিটার পানিতে ২ গ্রাম Dithane অথবা ১ মিলি Amistar Top মিশিয়ে স্প্রে করুন।"
    },
    'Potato___Late_blight': {
        "cause": "Phytophthora infestans ছত্রাক",
        "solution": "স্যাঁতস্যাঁতে আবহাওয়ায় আগাম সতর্কতা নিন। আক্রান্ত গাছ সরিয়ে ফেলুন।",
        "medicine": "Secure 600 WG / Ridomil Gold",
        "dosage": "২ গ্রাম প্রতি লিটার পানিতে মিশিয়ে ৫–৭ দিন পর পর স্প্রে করুন।"
    },
    'Potato___healthy': {
        "cause": "কোন রোগ নেই",
        "solution": "আলু গাছ সুস্থ। নিয়মিত পরিচর্যা বজায় রাখুন।",
        "medicine": "প্রযোজ্য নয়",
        "dosage": "-"
    },

    # ================= TOMATO (missing core ones) =================
    'Tomato___Early_blight': {
        "cause": "Alternaria solani ছত্রাক",
        "solution": "আক্রান্ত পাতা অপসারণ করুন। ফসল পর্যায়ক্রম অনুসরণ করুন।",
        "medicine": "Score 250 EC / Dithane M-45",
        "dosage": "০.৫ মিলি Score অথবা ২ গ্রাম Dithane প্রতি লিটার পানিতে মিশিয়ে স্প্রে করুন।"
    },
    'Tomato___Late_blight': {
        "cause": "Phytophthora infestans ছত্রাক",
        "solution": "আর্দ্র আবহাওয়ায় দ্রুত ব্যবস্থা নিন। আক্রান্ত গাছ সরান।",
        "medicine": "Acrobat MZ / Dithane M-45",
        "dosage": "২ গ্রাম প্রতি লিটার পানিতে মিশিয়ে স্প্রে করুন।"
    },
    'Tomato___Bacterial_spot': {
        "cause": "Xanthomonas ব্যাকটেরিয়া",
        "solution": "পাতায় পানি জমতে দেবেন না। পরিষ্কার বীজ ব্যবহার করুন।",
        "medicine": "Kocide 3000 / Copper Oxychloride",
        "dosage": "২–২.৫ গ্রাম প্রতি লিটার পানিতে মিশিয়ে স্প্রে করুন।"
    },
    'Tomato___Leaf_Mold': {
        "cause": "Passalora fulva ছত্রাক",
        "solution": "গ্রিনহাউস বা জমিতে বাতাস চলাচল বাড়ান।",
        "medicine": "Nativo 75 WG",
        "dosage": "০.৬ গ্রাম প্রতি লিটার পানিতে মিশিয়ে বিকেলে স্প্রে করুন।"
    }

}


# Fallback Generic Remedies
GENERIC_REMEDIES = {
    'healthy': "আপনার ফসল সুস্থ ও সবল আছে। নিয়মিত পরিচর্যা ও পর্যবেক্ষণ চালিয়ে যান।",
    'fungal': "ম্যানকোজেব জাতীয় ছত্রাকনাশক (যেমন: Dithane M-45) ২ গ্রাম/লিটার হারে স্প্রে করুন।",
    'bacterial': "কপার অক্সিক্লোরাইড জাতীয় বালাইনাশক ব্যবহার করুন।",
    'viral': "ভাইরাস আক্রান্ত গাছ তুলে মাটিতে পুঁতে ফেলুন এবং বাহক পোকা দমনে ইমিডাক্লোপ্রিড স্প্রে করুন।"
}

# Advanced Crop Preferences for Dynamic Reasoning
CROP_PREFERENCES = {
    'Rice': {
        'soil': ['Clay', 'Silty Clay', 'Clay Loam'],
        'ph_min': 5.5, 'ph_max': 8.0, 'water': 'High',
        'desc': 'কাদামাটি ও প্রচুর পানি ধান চাষের জন্য অপরিহার্য।'
    },

    'Wheat': {
        'soil': ['Loamy', 'Clay Loam', 'Silt'],
        'ph_min': 6.0, 'ph_max': 7.5, 'water': 'Medium',
        'desc': 'দোআঁশ ও পলি মাটি গম চাষের জন্য উপযোগী।'
    },

    'Jute': {
        'soil': ['Sandy Loam', 'Clay Loam', 'Silt'],
        'ph_min': 5.0, 'ph_max': 8.0, 'water': 'High',
        'desc': 'পলি ও দোআঁশ মাটি পাট চাষের জন্য আদর্শ।'
    },

    'Potato': {
        'soil': ['Sandy Loam', 'Loamy'],
        'ph_min': 4.8, 'ph_max': 6.5, 'water': 'Medium',
        'desc': 'ঝুরঝুরে বেলে দোআঁশ মাটি আলুর জন্য উপযুক্ত।'
    },

    'Onion': {
        'soil': ['Sandy Loam', 'Silty Loam'],
        'ph_min': 6.0, 'ph_max': 7.5, 'water': 'Medium',
        'desc': 'পেঁয়াজের জন্য পানি নিষ্কাশন ব্যবস্থা ভালো হতে হবে।'
    },

    'Garlic': {
        'soil': ['Loamy', 'Sandy Loam'],
        'ph_min': 6.0, 'ph_max': 7.0, 'water': 'Medium',
        'desc': 'রসুন চাষে ঝুরঝুরে দোআঁশ মাটি প্রয়োজন।'
    },

    'Lentil': {
        'soil': ['Loamy', 'Clay Loam'],
        'ph_min': 6.0, 'ph_max': 7.0, 'water': 'Low',
        'desc': 'ডাল শস্য কম পানিতে ভালো ফলন দেয়।'
    },

    'Mustard': {
        'soil': ['Sandy Loam', 'Clay Loam'],
        'ph_min': 6.0, 'ph_max': 7.5, 'water': 'Low',
        'desc': 'সরিষা কম সেচেও ভালো জন্মে।'
    },

    'Tomato': {
        'soil': ['Loamy', 'Sandy Loam'],
        'ph_min': 6.0, 'ph_max': 7.0, 'water': 'Medium',
        'desc': 'উর্বর দোআঁশ মাটি টমেটোর জন্য উপযোগী।'
    },

    'Eggplant': {
        'soil': ['Loamy', 'Clay Loam', 'Sandy Loam'],
        'ph_min': 5.5, 'ph_max': 7.0, 'water': 'High',
        'desc': 'বেগুনের জন্য নিয়মিত সেচ ও জৈব সার দরকার।'
    },

    'Chili': {
        'soil': ['Sandy Loam', 'Loamy'],
        'ph_min': 6.0, 'ph_max': 7.0, 'water': 'Medium',
        'desc': 'মরিচ গাছের গোড়ায় পানি জমা ক্ষতিকর।'
    },

    'Chilli': {
        'soil': ['Sandy Loam', 'Loamy'],
        'ph_min': 6.0, 'ph_max': 7.0, 'water': 'Medium',
        'desc': 'মরিচ চাষে পানি নিষ্কাশন জরুরি।'
    },

    'Cabbage': {
        'soil': ['Clay Loam', 'Sandy Loam'],
        'ph_min': 6.0, 'ph_max': 7.5, 'water': 'High',
        'desc': 'আর্দ্র মাটি বাঁধাকপির জন্য ভালো।'
    },

    'Cauliflower': {
        'soil': ['Loamy', 'Sandy Loam'],
        'ph_min': 6.0, 'ph_max': 7.0, 'water': 'High',
        'desc': 'ফুলকপির জন্য মাটির আর্দ্রতা গুরুত্বপূর্ণ।'
    },

    'Cucumber': {
        'soil': ['Sandy Loam', 'Loamy'],
        'ph_min': 6.0, 'ph_max': 7.0, 'water': 'High',
        'desc': 'শসা চাষে নিয়মিত সেচ প্রয়োজন।'
    },

    'Pumpkin': {
        'soil': ['Sandy Loam', 'Loamy'],
        'ph_min': 5.5, 'ph_max': 7.5, 'water': 'Medium',
        'desc': 'কুমড়া পানি জমে না এমন মাটিতে ভালো হয়।'
    },

    'Bitter Gourd': {
        'soil': ['Sandy Loam', 'Loamy'],
        'ph_min': 6.0, 'ph_max': 7.0, 'water': 'Medium',
        'desc': 'করলা চাষে মাচা পদ্ধতি কার্যকর।'
    },

    'Bottle Gourd': {
        'soil': ['Loamy', 'Sandy Loam'],
        'ph_min': 6.0, 'ph_max': 7.5, 'water': 'Medium',
        'desc': 'লাউ গভীর ও উর্বর মাটিতে ভালো জন্মে।'
    },

    'Okra': {
        'soil': ['Sandy Loam', 'Clay Loam'],
        'ph_min': 6.0, 'ph_max': 6.8, 'water': 'Medium',
        'desc': 'ঢেঁড়স উষ্ণ আবহাওয়ায় ভালো ফলন দেয়।'
    },

    'Spinach': {
        'soil': ['Loamy', 'Sandy Loam', 'Nitrogenous'],
        'ph_min': 6.0, 'ph_max': 7.0, 'water': 'Medium',
        'desc': 'পালং শাকে নাইট্রোজেন সমৃদ্ধ মাটি উপকারী।'
    },

    'Maize': {
        'soil': ['Loamy', 'Sandy Loam'],
        'ph_min': 5.5, 'ph_max': 7.5, 'water': 'Medium',
        'desc': 'ভুট্টা চাষে পানি জমে থাকা এড়িয়ে চলুন।'
    },

    'Sugarcane': {
        'soil': ['Loamy', 'Clay Loam'],
        'ph_min': 6.5, 'ph_max': 7.5, 'water': 'High',
        'desc': 'আখ চাষে দীর্ঘমেয়াদি আর্দ্রতা প্রয়োজন।'
    },

    'Tea': {
        'soil': ['Sandy Loam'],
        'ph_min': 4.5, 'ph_max': 5.8, 'water': 'High',
        'desc': 'চা চাষে বেলে দোআঁশ ও ভালো নিষ্কাশন দরকার।'
    },

    'Mango': {
        'soil': ['Loamy'],
        'ph_min': 5.5, 'ph_max': 7.5, 'water': 'Medium',
        'desc': 'আম বাগানের জন্য দোআঁশ মাটি উপযোগী।'
    },

    'Banana': {
        'soil': ['Loamy'],
        'ph_min': 6.0, 'ph_max': 7.5, 'water': 'High',
        'desc': 'কলা আর্দ্র ও উর্বর দোআঁশ মাটিতে ভালো হয়।'
    },

    'Jackfruit': {
        'soil': ['Loamy', 'Sandy Loam'],
        'ph_min': 6.0, 'ph_max': 7.5, 'water': 'Medium',
        'desc': 'কাঁঠাল পানি জমে না এমন জমিতে ভালো হয়।'
    },

    'Papaya': {
        'soil': ['Loamy', 'Sandy Loam'],
        'ph_min': 6.0, 'ph_max': 7.0, 'water': 'Medium',
        'desc': 'পেঁপে চাষে ভালো নিষ্কাশন খুব জরুরি।'
    },

    'Guava': {
        'soil': ['Loamy'],
        'ph_min': 4.5, 'ph_max': 8.2, 'water': 'Medium',
        'desc': 'পেয়ারা বিভিন্ন pH-এর মাটিতে মানিয়ে নেয়।'
    },

    'Lychee': {
        'soil': ['Loamy'],
        'ph_min': 6.0, 'ph_max': 7.0, 'water': 'High',
        'desc': 'লিচু চাষে উর্বর দোআঁশ মাটি প্রয়োজন।'
    },

    'Pineapple': {
        'soil': ['Sandy Loam'],
        'ph_min': 4.5, 'ph_max': 6.0, 'water': 'Medium',
        'desc': 'আনারস অম্লীয় বেলে দোআঁশ মাটিতে ভালো হয়।'
    },

    'Cotton': {
        'soil': ['Black lava soil', 'Loamy'],
        'ph_min': 5.5, 'ph_max': 8.5, 'water': 'Low',
        'desc': 'তুলা চাষে কালো মাটি সবচেয়ে উপযোগী।'
    },

    'Gram': {
        'soil': ['Sandy Loam', 'Clay Loam'],
        'ph_min': 6.0, 'ph_max': 9.0, 'water': 'Low',
        'desc': 'ছোলা কম পানিতেও ভালো জন্মে।'
    }
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
    
# --- Price Prediction Helper (Refactored) ---
@st.cache_data(ttl=3600)
def predict_next_30_days(district, crop, _df):
    """
    Predicts daily prices for the next 30 days for a given district and crop using RandomForest.
    Returns:
        avg_predicted_price (float): Mean of the 30-day predictions.
        future_data (pd.DataFrame): DataFrame containing dates, predicted prices, and confidence intervals.
        historical_data (pd.DataFrame): The filtered historical data used for training (for plotting).
    """
    # Filter data
    filtered_df = _df[(_df['District_Name'] == district) & (_df['Crop_Name'] == crop)].sort_values('Price_Date')
    
    if len(filtered_df) < 10:
        return None, None, None

    # Feature Engineering
    filtered_df['Date_Ordinal'] = filtered_df['Price_Date'].map(datetime.datetime.toordinal)
    filtered_df['Month'] = filtered_df['Price_Date'].dt.month
    filtered_df['Week'] = filtered_df['Price_Date'].dt.isocalendar().week
    filtered_df['Year'] = filtered_df['Price_Date'].dt.year
    
    X = filtered_df[['Date_Ordinal', 'Month', 'Week', 'Year']]
    y = filtered_df['Price_Tk_kg']
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Prepare Future Data
    last_date = filtered_df['Price_Date'].max()
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]
    future_data = pd.DataFrame({'Price_Date': future_dates})
    future_data['Date_Ordinal'] = future_data['Price_Date'].map(datetime.datetime.toordinal)
    future_data['Month'] = future_data['Price_Date'].dt.month
    future_data['Week'] = future_data['Price_Date'].dt.isocalendar().week
    future_data['Year'] = future_data['Price_Date'].dt.year
    
    # Get predictions
    predictions = model.predict(future_data[['Date_Ordinal', 'Month', 'Week', 'Year']])
    
    # Confidence Intervals
    tree_predictions = np.array([tree.predict(future_data[['Date_Ordinal', 'Month', 'Week', 'Year']].values) for tree in model.estimators_])
    std_predictions = tree_predictions.std(axis=0)
    
    future_data['Predicted_Price'] = predictions
    future_data['Upper_Bound'] = predictions + 1.96 * std_predictions
    future_data['Lower_Bound'] = predictions - 1.96 * std_predictions
    
    avg_predicted_price = predictions.mean()
    
    return avg_predicted_price, future_data, filtered_df

def get_crop_reasoning(soil_record, crop, yield_val):
    """
    Generate diverse, crop-specific reasoning based on soil conditions.
    """
    import random

    soil_type = soil_record['Soil_Type']
    ph = soil_record['pH_Level']
    nitrogen = soil_record['Nitrogen_Content_kg_ha']
    organic = soil_record['Organic_Matter_Percent']

    prefs = CROP_PREFERENCES.get(crop, {})
    reasons = []

    soil_bn = translate_bn(soil_type, soil_translation)
    crop_bn = translate_bn(crop, crop_translation)

    # 1. Intro variations
    intros = [
        f"এই অঞ্চলের **{soil_bn}** {crop_bn} চাষের জন্য",
        f"ঐতিহাসিকভাবে এখানে {crop_bn} ভালো হয় কারণ এখানকার **{soil_bn}**",
        f"উপাত্ত বিশ্লেষণ অনুযায়ী, **{soil_bn}** থাকায় এই এলাকা {crop_bn} উৎপাদনে"
    ]

    # 2. Soil Suitability Check
    is_soil_ideal = False
    if prefs and 'soil' in prefs:
        soil_match = any(
            s.lower() in soil_type.lower() or soil_type.lower() in s.lower()
            for s in prefs['soil']
        )
        if soil_match:
            reasons.append(f"{random.choice(intros)} অত্যন্ত উপযোগী।")
            is_soil_ideal = True
        else:
            reasons.append(f"{random.choice(intros)} মোটামুটি মানানসই (বিশেষ ব্যবস্থাপনা প্রয়োজন)।")
    else:
        reasons.append(f"এই এলাকার মাটি ও আবহাওয়া {crop_bn} চাষের জন্য গ্রহণযোগ্য।")

    # 3. Crop-specific insight
    if prefs.get('desc'):
        reasons.append(f"💡 **বিশেষ নোট:** {prefs['desc']}")

    # 4. Nitrogen Analysis
    high_n_crops = ['Rice', 'Maize', 'Wheat', 'Sugarcane', 'Tea', 'Mustard']
    legumes = ['Lentil', 'Gram']

    if nitrogen > 150:
        if crop in high_n_crops:
            reasons.append(
                f"✅ মাটিতে পর্যাপ্ত নাইট্রোজেন ({nitrogen:.1f} kg/ha) আছে, যা এই ফসলের বৃদ্ধিতে সহায়ক।"
            )
        elif crop in legumes:
            reasons.append(
                f"⚠️ নাইট্রোজেনের পরিমাণ বেশি ({nitrogen:.1f} kg/ha); অতিরিক্ত ইউরিয়া প্রয়োগ এড়িয়ে চলুন।"
            )
        else:
            reasons.append(
                f"ℹ️ নাইট্রোজেনের মাত্রা ({nitrogen:.1f} kg/ha) মাঝারি–উচ্চ, সুষম সার ব্যবস্থাপনা দরকার।"
            )

    elif nitrogen < 100:
        if crop in legumes:
            reasons.append(
                "✅ নাইট্রোজেন কম হলেও সমস্যা নয়, কারণ এটি ডালজাতীয় ফসল এবং নিজেই নাইট্রোজেন স্থির করতে পারে।"
            )
        else:
            reasons.append(
                f"⚠️ নাইট্রোজেনের ঘাটতি ({nitrogen:.1f} kg/ha) রয়েছে; ইউরিয়া বা জৈব সার প্রয়োগ প্রয়োজন।"
            )

    # 5. pH Analysis
    if prefs:
        min_ph = prefs.get('ph_min', 5.5)
        max_ph = prefs.get('ph_max', 7.5)

        if min_ph <= ph <= max_ph:
            reasons.append(f"✅ মাটির pH ({ph:.1f}) এই ফসলের জন্য আদর্শ।")
        elif ph < min_ph:
            reasons.append(
                f"⚠️ মাটি বেশি অম্লীয় (pH {ph:.1f}); চুন প্রয়োগ করলে ফলন বাড়তে পারে।"
            )
        else:
            reasons.append(
                f"⚠️ মাটি কিছুটা ক্ষারীয় (pH {ph:.1f}); জৈব সার ও জিপসাম উপকারী হতে পারে।"
            )

    # 6. Water / Irrigation Logic
    water_req = prefs.get('water', 'Medium')

    if water_req == 'High':
        if 'Clay' in soil_type:
            reasons.append(
                "💧 **সেচ সুবিধা:** কাদামাটি পানি ধরে রাখতে পারে, যা এই ফসলের জন্য বড় সুবিধা।"
            )
        elif 'Sandy' in soil_type:
            reasons.append(
                "⚠️ **সেচ সতর্কতা:** বেলে মাটি পানি ধরে রাখতে পারে না; ঘন ঘন সেচ প্রয়োজন।"
            )
        else:
            reasons.append(
                "💧 **সেচ পরামর্শ:** নিয়মিত ও পরিকল্পিত সেচ নিশ্চিত করুন।"
            )

    elif water_req == 'Low' and 'Clay' in soil_type:
        reasons.append(
            "⚠️ **নিষ্কাশন সতর্কতা:** এই ফসলের কম পানি লাগে, কাদামাটিতে পানি জমে গেলে ক্ষতি হতে পারে।"
        )

    # 7. Yield Projection
    yield_desc = "খুবই ভালো" if yield_val > 40 else "সন্তোষজনক"
    reasons.append(
        f"📈 **প্রত্যাশিত ফলন:** হেক্টর প্রতি প্রায় **{to_bengali_number(f'{yield_val:.1f}')}** কুইন্টাল, যা {yield_desc}।"
    )

    return "\n\n".join(reasons)


# --- Top Menu (Mobile Optimized) ---
# --- Menu Constants ---
MENU_OPTIONS = ["📊 মূল্য পূর্বাভাস (এআই)", "💰 সেরা বাজার খুঁজুন", "🌱 মাটি ও ফসল পরামর্শদাতা", "🦠 ফসল বিষাক্তি পরিচিতি", "📊 এগ্রি-ফাইন্যান্স ও লাভ রিপোর্ট"]
MENU_ICONS = ["graph-up-arrow", "shop", "flower1", "bug", "cash-coin"]

# --- State Management for Menu Sync ---
if 'current_module' not in st.session_state:
    st.session_state.current_module = MENU_OPTIONS[0]

def update_from_sidebar():
    st.session_state.current_module = st.session_state.nav_sidebar


def process_voice_command(voice_text, district_names=None, crop_names=None, district_key=None, crop_key=None):
    """
    Process voice text to find district and crop names.
    Updates session state keys if matches are found.
    Returns: True if any update occurred, False otherwise.
    """
    if not voice_text: return False
    
    updated = False
    
    # Check for District
    if district_names and district_key:
        found_dist = None
        # Try finding exact match first, then partial
        for dist_bn in district_names:
            if dist_bn in voice_text:
                found_dist = dist_bn
                break
        
        if found_dist:
            # Only update if different
            current_val = st.session_state.get(district_key)
            if current_val != found_dist:
                st.session_state[district_key] = found_dist
                st.toast(f"✅ জেলা শনাক্ত হয়েছে: {found_dist}")
                updated = True
    
    # Check for Crop
    if crop_names and crop_key:
        found_crop = None
        for crop_bn in crop_names:
            if crop_bn in voice_text:
                found_crop = crop_bn
                break
        
        if found_crop:
            current_val = st.session_state.get(crop_key)
            # Find the option that matches this crop name (since selectbox might use format_func, but key stores the value)
            # Assuming the key stores the Bengali name directly for these specific selectboxes as per app structure
            if current_val != found_crop:
                 st.session_state[crop_key] = found_crop
                 st.toast(f"✅ ফসল শনাক্ত হয়েছে: {found_crop}")
                 updated = True

    return updated
    
def update_from_mobile(key):
    st.session_state.current_module = st.session_state[key]

# --- Sidebar (Desktop) ---
st.sidebar.markdown("**এআই চালিত কৃষি বুদ্ধিমত্তা**")
# Ensure the sidebar reflects the current state (e.g. if changed from mobile)
st.sidebar.radio(
    "মডিউল নির্বাচন করুন:", 
    MENU_OPTIONS,
    index=MENU_OPTIONS.index(st.session_state.current_module) if st.session_state.current_module in MENU_OPTIONS else 0,
    key="nav_sidebar",
    on_change=update_from_sidebar
)

# --- Top Menu (Mobile Optimized) ---
# Inject marker for CSS to find
st.markdown('<div id="mobile-menu-marker"></div>', unsafe_allow_html=True)

with st.expander("☰ মেনু নির্বাচন (Menu)", expanded=False):
    # Ensure options menu reflects current state
    selected = option_menu(
        menu_title=None, 
        options=MENU_OPTIONS, 
        icons=MENU_ICONS, 
        menu_icon="cast", 
        default_index=MENU_OPTIONS.index(st.session_state.current_module) if st.session_state.current_module in MENU_OPTIONS else 0,
        orientation="vertical",
        key="nav_mobile",
        on_change=update_from_mobile,
        styles={
            "container": {"padding": "0!important", "background-color": "#ffffff"},
            "icon": {"color": "#FF8C00", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#11998e", "color": "white"},
        }
    )

# Use the synced state variable for page logic
menu = st.session_state.current_module

# -----------------------------------------------------------------------------
# MODULE 1: AI PRICE FORECASTING
# -----------------------------------------------------------------------------
if menu == "📊 মূল্য পূর্বাভাস (এআই)":
    st.markdown("### মেশিন লার্নিং ব্যবহার করে ৩০ দিনের আগাম মূল্যের পূর্বাভাস।")
    
    # --- GEOLOCATION & PIN-POINT WEATHER SECTION ---
    
    # Session Variables for Location
    if 'user_lat' not in st.session_state: st.session_state.user_lat = None
    if 'user_lon' not in st.session_state: st.session_state.user_lon = None
    if 'detected_city' not in st.session_state: st.session_state.detected_city = "Unknown Location"

    # Geolocation Button
    c_geo1, c_geo2 = st.columns([1, 3])
    with c_geo1:
        if st.button("📍 আমার বর্তমান অবস্থান ব্যবহার করুন"):
            st.session_state.finding_location = True
    
    if st.session_state.get('finding_location', False):
        with st.spinner("GPS অবস্থান নির্ণয় করা হচ্ছে (অনুগ্রহ করে অনুমতি দিন)..."):
            try:
                # Use HTML5 Geolocation API for Pin-point accuracy
                # Fix: explicit JSON for success, and explicit payload for error.
                loc_data = streamlit_js_eval(
                    js_expressions='new Promise((resolve) => navigator.geolocation.getCurrentPosition(p => resolve({coords: {latitude: p.coords.latitude, longitude: p.coords.longitude}}), e => resolve({error: true})))', 
                    key='geo_gps_fetch'
                )
                
                if loc_data:
                    if 'coords' in loc_data:
                        lat = loc_data['coords']['latitude']
                        lon = loc_data['coords']['longitude']
                        
                        st.session_state.user_lat = float(lat)
                        st.session_state.user_lon = float(lon)
                        st.session_state.detected_city = "আপনার অবস্থান" 
                        st.success("✅ অবস্থান শনাক্ত হয়েছে!")
                        st.session_state.finding_location = False
                        time.sleep(1)
                        st.rerun()
                    elif 'error' in loc_data:
                         st.warning("⚠️ GPS অনুমতি পাওয়া যায়নি বা সমস্যা হয়েছে।")
                         st.session_state.finding_location = False
                else:
                    # loc_data is None -> JS is still executing or not ready. Do nothing.
                    pass
            except Exception as e:
                st.error("অবস্থান নির্ণয়ে সমস্যা হয়েছে।")
                st.session_state.finding_location = False

    # --- DISTRICT & SESSION SETUP ---
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

    # --- REAL-TIME WEATHER ALERT LOGIC ---
    weather_api_key = st.secrets.get("WEATHER_API_KEY") or os.getenv("WEATHER_API_KEY")
    
    current_w = None
    forecast_w = None
    location_label = ""
    is_gps = False
    
    # 1. Try GPS Location
    if st.session_state.user_lat and st.session_state.user_lon and weather_api_key:
        lat = st.session_state.user_lat
        lon = st.session_state.user_lon
        current_w = get_weather_by_coords(lat, lon, weather_api_key)
        forecast_w = get_forecast_by_coords(lat, lon, weather_api_key)
        
        # Use city name from API if available
        api_city = current_w.get('name') if current_w else None
        display_city = api_city if api_city else st.session_state.detected_city
        location_label = f"{display_city} (GPS)"
        is_gps = True
        
    # 2. Fallback to Selected District
    elif weather_api_key and 'selected_district_val' in st.session_state:
        # Get English name
        dist_bn = st.session_state.selected_district_val
        # Find key by value
        dist_eng = [k for k, v in district_display.items() if v == dist_bn]
        if dist_eng:
            search_city = dist_eng[0]
            current_w = get_weather_data(search_city, weather_api_key)
            forecast_w = get_weather_forecast(search_city, weather_api_key)
            location_label = f"{dist_bn} (District)"
            
    # 3. Process & Display Weather
    if current_w:
        # Current Data
        temp = current_w['main']['temp']
        humidity = current_w['main']['humidity']
        desc = current_w['weather'][0]['description'].title()
        icon = current_w['weather'][0]['icon']
            
        # Analyze Forecast
        rain_prob = 0
        is_rain_likely = False
            
        if forecast_w:
            for item in forecast_w['list'][:8]:
                pop = item.get('pop', 0)
                if pop > 0.7:
                    is_rain_likely = True
                    rain_prob = int(pop * 100)
                    break
            
        # Generate Advisory
        alert_color = "#4caf50" # Green
        alert_msg = "✅ আবহাওয়া চাষাবাদের জন্য অনুকূল।"
            
        if is_rain_likely:
            alert_color = "#ff4b4b" # Red
            alert_msg = f"⚠️ **সতর্কতা:** আগামী ২৪ ঘন্টায় বৃষ্টির সম্ভাবনা {rain_prob}%। জমিতে সার বা কীটনাশক দেবেন না।"
        elif temp > 36:
            alert_color = "#ffa726" # Orange
            alert_msg = "☀️ **তাপপ্রবাহ:** অতিরিক্ত তাপমাত্রা। ফসলে সেচ নিশ্চিত করুন।"
            
        # Extract additional details
        feels_like = current_w['main']['feels_like']
        wind_speed = current_w['wind']['speed']
        pressure = current_w['main']['pressure']

        # Display Card with Expanded Info
        st.markdown(f"""
        <div style="background-color: #f8f9fa; border-radius: 12px; padding: 15px; border: 1px solid #ddd; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <img src="http://openweathermap.org/img/wn/{icon}@2x.png" width="70">
                    <div>
                        <h3 style="margin: 0; color: #333;">{temp:.1f}°C</h3>
                        <p style="margin: 0; color: #666; font-size: 14px;">{desc} | অনুভূতি: {feels_like:.1f}°C</p>
                        <p style="margin: 0; color: #666; font-size: 13px;">💧 আর্দ্রতা: {humidity}% | 🌬️ বাতাস: {wind_speed} m/s | 🌡️ চাপ: {pressure} hPa</p>
                        <small style="color: #888;">📍 {location_label}</small>
                    </div>
                </div>
                <div style="background-color: {alert_color}; color: white; padding: 10px 20px; border-radius: 8px; margin-top: 10px; text-align: right;">
                    {alert_msg}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
            
        # Map (Only for GPS)
        if is_gps:
            st.markdown("**🗺️ আপনার জমির অবস্থান (OpenStreetMap):**")
            map_data = pd.DataFrame({'lat': [st.session_state.user_lat], 'lon': [st.session_state.user_lon]})
            st.map(map_data, zoom=12, use_container_width=True)
            
            # Auto-Sync GPS City to District Selection (One-time sync per GPS fetch)
            if 'sync_done' not in st.session_state: st.session_state.sync_done = False
            
            # Use detected city from IP/weather data if available
            detected_city_name = current_w.get('name', st.session_state.detected_city)
            
            if detected_city_name and not st.session_state.sync_done:
                match_found_bn = None
                
                # Normalize for matching
                search_name = detected_city_name.lower().strip()
                
                # Check 1: Direct/Case-insensitive matching with keys
                for d_eng, d_bn in district_translation.items():
                    if d_eng.lower() in search_name or search_name in d_eng.lower():
                        match_found_bn = d_bn
                        break
                
                # Update Session State if match found
                if match_found_bn and 'selected_district_val' in st.session_state:
                    if st.session_state.selected_district_val != match_found_bn:
                        st.session_state.selected_district_val = match_found_bn
                        st.session_state.sync_done = True # Prevent infinite loops
                        st.toast(f"📍 জেলা স্বয়ংক্রিয়ভাবে নির্বাচিত: {match_found_bn}")
                        time.sleep(0.5)
                        st.rerun()



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
            # Check if this voice command was already processed
            prev_text = st.session_state.get('last_voice_text', "")
            if voice_text != prev_text:
                updated = process_voice_command(
                    voice_text, 
                    district_names=district_options_list, 
                    district_key='selected_district_val'
                )
                
                st.session_state.last_voice_text = voice_text # Mark processed
                
                if updated:
                    time.sleep(1)
                    st.rerun()
                elif not any(d in voice_text for d in district_options_list):
                    st.toast("⚠️ কোনো জেলা পাওয়া যায়নি", icon="⚠️")
    
    # Legacy Geolocation logic removed


    st.divider()

    # Inputs
    col1, col2 = st.columns(2)
    def reset_gps_state():
        if 'user_lat' in st.session_state: st.session_state.user_lat = None
        if 'user_lon' in st.session_state: st.session_state.user_lon = None
        if 'sync_done' in st.session_state: st.session_state.sync_done = False # Reset sync flag

        
    with col1:
        selected_district_bn = st.selectbox("📍 জেলা নির্বাচন করুন", options=district_options_list, key='selected_district_val', on_change=reset_gps_state)
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
        f_data = get_weather_forecast(selected_district, weather_api_key)
        
        if w_data:
            main = w_data['main']
            weather_desc = w_data['weather'][0]['description']
            icon_code = w_data['weather'][0]['icon']
            weather_icon_url = f"http://openweathermap.org/img/wn/{icon_code}@2x.png"
            
            # --- DISASTER ALERT LOGIC (Feature 1) ---
            alert_msg = ""
            alert_color = "#e3f2fd" # Default blue
            alert_icon = "✅"
            show_red_alert = False

            # Check Forecast for Rain
            rain_prob = 0
            if f_data:
                # Check next 24 hours (8 * 3hr intervals)
                for item in f_data['list'][:8]:
                    if 'rain' in item:
                        # rain probability is not directly given in standard free API, 
                        # but 'pop' (probability of precipitation) is available in 2.5/forecast
                        pop = item.get('pop', 0)
                        if pop > rain_prob: rain_prob = pop
            
            # Artificial Logic for Demo if 'pop' unavailable or 0 (remove in prod if needed)
            if 'rain' in weather_desc.lower(): 
                rain_prob = 0.8
            
            if rain_prob > 0.7:
                show_red_alert = True
                alert_msg = "⚠️ আগামীকাল বৃষ্টির সম্ভাবনা আছে, আজ সেচ বা সার দেওয়া থেকে বিরত থাকুন।"
                alert_color = "#ffebee" # Red background
                alert_icon = "⛈️"
            elif main['temp'] > 35:
                alert_msg = "☀️ সতর্কতা: অতিরিক্ত তাপমাত্রা। জমিতে পর্যাপ্ত সেচ নিশ্চিত করুন।"
                alert_color = "#fff3e0" # Orange
                alert_icon = "🔥"
            elif main['humidity'] > 85:
                 alert_msg = "💧 সতর্কতা: উচ্চ আর্দ্রতা। ছত্রাকজনিত রোগের ঝুঁকি বেশি।"
                 alert_color = "#e0f2f1"
                 alert_icon = "💧"
            else:
                 alert_msg = "✅ আবহাওয়া চাষাবাদের অনুকূল।"

            # Display Weather Dashboard
            # Display Weather Dashboard
            # Legacy Weather UI removed/commented out as per user request to use the new card
            # Keeping the variable selected_district for prediction logic below
            pass
    
    # Analysis & Prediction
    # Analysis & Prediction
    avg_pred_price, future_df, hist_df = predict_next_30_days(selected_district, selected_crop, price_df)

    if future_df is not None:
        # Plot with confidence intervals
        st.subheader(f"মূল্য প্রবণতা: {translate_bn(selected_crop, crop_translation)}")
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=hist_df['Price_Date'], 
            y=hist_df['Price_Tk_kg'], 
            mode='lines', 
            name='ঐতিহাসিক', 
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Predicted data
        fig.add_trace(go.Scatter(
            x=future_df['Price_Date'], 
            y=future_df['Predicted_Price'], 
            mode='lines', 
            name='পূর্বাভাস', 
            line=dict(color='#00cc96', width=2)
        ))
        
        # Confidence interval upper bound
        fig.add_trace(go.Scatter(
            x=future_df['Price_Date'],
            y=future_df['Upper_Bound'],
            mode='lines',
            name='উর্ধ্ব সীমা',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Confidence interval lower bound with fill
        fig.add_trace(go.Scatter(
            x=future_df['Price_Date'],
            y=future_df['Lower_Bound'],
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

        current_price = hist_df.iloc[-1]['Price_Tk_kg']
        avg_price = avg_pred_price
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
    
    # Session state for target crop
    if 'target_crop_bn' not in st.session_state:
        st.session_state.target_crop_bn = list(all_crops_display.values())[0]

    # Voice Input
    c1, c2 = st.columns([1, 4])
    with c1:
        audio = mic_recorder(start_prompt="🎤 বলুন", stop_prompt="🛑 থামুন", key='rec_best_market', format="wav", use_container_width=True)
    
    if audio:
        voice_text = voice_to_text(audio['bytes'])
        if voice_text:
            st.success(f"🗣️: **'{voice_text}'**")
            updated = process_voice_command(
                voice_text, 
                crop_names=list(all_crops_display.values()), 
                crop_key='target_crop_bn'
            )
            if updated:
                time.sleep(0.5)
                st.rerun()
            elif not any(c in voice_text for c in list(all_crops_display.values())):
                 st.toast("⚠️ কোনো ফসল পাওয়া যায়নি", icon="⚠️")

    target_crop_bn = st.selectbox("🔍 ফসল নির্বাচন করুন", options=list(all_crops_display.values()), key='target_crop_bn')
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

    if 'target_district_bn_soil' not in st.session_state:
        st.session_state.target_district_bn_soil = list(soil_district_display.values())[default_idx]

    # Voice Input
    c1, c2 = st.columns([1, 4])
    with c1:
        audio = mic_recorder(start_prompt="🎤 বলুন", stop_prompt="🛑 থামুন", key='rec_soil', format="wav", use_container_width=True)
    
    if audio:
        voice_text = voice_to_text(audio['bytes'])
        if voice_text:
            st.success(f"🗣️: **'{voice_text}'**")
            updated = process_voice_command(
                voice_text, 
                district_names=list(soil_district_display.values()), 
                district_key='target_district_bn_soil'
            )
            if updated:
                time.sleep(0.5)
                st.rerun()
            elif not any(d in voice_text for d in list(soil_district_display.values())):
                 st.toast("⚠️ কোনো জেলা পাওয়া যায়নি", icon="⚠️")

    target_district_bn = st.selectbox("📍 অবস্থান নির্বাচন করুন", options=list(soil_district_display.values()), index=default_idx, key='target_district_bn_soil')
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
    # st.markdown("Upload a photo of your crop leaf for AI analysis (99.2% accuracy on global dataset). Note: This is for guidance only—consult local agri experts for confirmation.")
    st.markdown(
    "AI বিশ্লেষণের জন্য আপনার ফসলের পাতার একটি ছবি আপলোড করুন "
    "(গ্লোবাল ডেটাসেটে ৯৯.২% নির্ভুলতা)। "
    "নোট: এটি শুধুমাত্র প্রাথমিক দিকনির্দেশনার জন্য—চূড়ান্ত সিদ্ধান্তের আগে "
    "স্থানীয় কৃষি বিশেষজ্ঞের পরামর্শ নিন।"
)



    model = load_plant_model()
    if not model:
        st.error("মডেল লোড হতে সমস্যা হয়েছে। ইন্টারনেট সংযোগ চেক করুন।")
    
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

        # Step 1: Leaf Detection Filter
        is_leaf, msg = is_likely_leaf(image)
        if not is_leaf:
            st.warning(msg)
            st.info("টিপস: ছবিটি উজ্জ্বল আলোতে তুলুন এবং ব্যাকগ্রাউন্ডে যেন পাতা থাকে তা নিশ্চিত করুন।")
            st.stop() # Stop further processing

        with st.spinner("রোগ নির্ণয় করা হচ্ছে..."):
            try:
                # Preprocess for (380x380)
                # 1. Resize
                img_resized = ImageOps.fit(image, (380, 380), Image.Resampling.LANCZOS)
                
                # 2. Convert to Array and Batch
                img_array = np.asarray(img_resized)
                img_batch = np.expand_dims(img_array, axis=0)
                
                # 3. Preprocess Input (Standard)
                img_preprocessed = preprocess_input(img_batch)

                # Inference
                probs = model.predict(img_preprocessed)
                
                # Get Prediction
                confidence_val = np.max(probs)
                pred_idx = np.argmax(probs)
                
                pred_class = CLASS_LABELS[pred_idx]
                conf_score = confidence_val * 100
            except Exception as e:
                st.error(f"Inference Error: {e}")
                st.stop()

        # Display Results
        bn_label = DISEASE_TRANSLATION.get(pred_class, pred_class)
        
        st.divider()
        st.subheader("ফলাফল:")
        
        # Result Badge
        is_healthy = "healthy" in pred_class.lower()
        if is_healthy:
            st.success(f"✅ **অবস্থা:** {bn_label}")
        else:
            st.error(f"⚠️ **শনাক্ত রোগ:** {bn_label}")
            
        # Confidence Bar
        st.write(f"**সঠিকতার হার:** {conf_score:.1f}%")
        st.progress(int(conf_score))
        
        # --- AI DOCTOR PRESCRIPTION (Feature 2) ---
        st.markdown("### 💊 এআই কৃষি ডাক্তার (Digital Prescription)")
        
        prescription = CROP_PRESCRIPTION_MAP.get(pred_class)
        
        if is_healthy:
             st.info(GENERIC_REMEDIES['healthy'])
        elif prescription:
            # Structured Prescription Card
            st.markdown(f"""
<div style="background-color: #f1f8e9; border: 2px solid #81c784; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
<h3 style="color: #2e7d32; margin-top: 0;">📋 ব্যবস্থাপত্র (Prescription)</h3>
<hr>
<div style="display: grid; grid-template-columns: 1fr; gap: 10px;">
<div>
<strong style="color: #e65100;">🔍 রোগের কারণ:</strong>
<span style="color: #333;">{prescription['cause']}</span>
</div>
<div>
<strong style="color: #1565c0;">🛡️ প্রতিকার/করণীয়:</strong>
<span style="color: #333;">{prescription['solution']}</span>
</div>
<div>
<strong style="color: #d32f2f;">💊 প্রস্তাবিত ঔষধ:</strong>
<span style="font-weight: bold; color: #d32f2f; background-color: #ffebee; padding: 2px 8px; border-radius: 4px;">{prescription['medicine']}</span>
</div>
<div>
<strong style="color: #43a047;">⚖️ মাত্রা (Dosage):</strong>
<span style="color: #333;">{prescription['dosage']}</span>
</div>
</div>
<div style="margin-top: 15px; font-size: 0.9em; color: #666; font-style: italic;">
* ঔষধ ব্যবহারের পূর্বে বোতলের গায়ের নির্দেশাবলী ভালো করে পড়ুন এবং সুরক্ষা পোশাক পরিধান করুন।
</div>
</div>
""", unsafe_allow_html=True)
        else:
            # Fallback for unconnected diseases
            fallback_remedy = GENERIC_REMEDIES['bacterial'] if 'bacterial' in pred_class.lower() \
                else (GENERIC_REMEDIES['viral'] if 'virus' in pred_class.lower() \
                else GENERIC_REMEDIES['fungal'])
            
            st.warning(f"⚠️ নির্দিষ্ট প্রেসক্রিপশন ডেটাবেসে নেই। সাধারণ পরামর্শ: {fallback_remedy}")
            
        # Remedy Section (Legacy) removed in favor of AI Doctor
            
        # Disclaimer
        with st.expander("⚠️ দাবিত্যাগ (Disclaimer)"):
            st.write("এই এআই মডেলটি সহায়ক টুল হিসেবে তৈরি। এটি ৯৯.২% নির্ভুল হলেও, চূড়ান্ত সিদ্ধান্তের জন্য সর্বদা কৃষি বিশেষজ্ঞের পরামর্শ নিন।")

# -----------------------------------------------------------------------------
# MODULE 5: AGRI-FINANCE & PROFIT REPORT (Feature 3)
# -----------------------------------------------------------------------------
elif menu == "📊 এগ্রি-ফাইন্যান্স ও লাভ রিপোর্ট":
    st.title("📊 এগ্রি-ফাইন্যান্স ও লাভ রিপোর্ট")
    st.markdown("### চাষাবাদের সম্ভাব্য আয় ও লোনের যোগ্যতা যাচাই করুন")
    st.divider()

    # 1. Inputs
    c1, c2 = st.columns(2)
    
    with c1:
        # District Selection
        district_list = sorted(price_df['District_Name'].unique())
        district_display = {dist: translate_bn(dist, district_translation) for dist in district_list}
        
        # Crop Selection
        all_crops = sorted(price_df['Crop_Name'].unique())
        all_crops_display = {crop: translate_bn(crop, crop_translation) for crop in all_crops}

        # Voice Input (PROCESSED BEFORE WIDGETS)
        audio = mic_recorder(start_prompt="🎤 বলুন", stop_prompt="🛑 থামুন", key='rec_calc', format="wav", use_container_width=True)
        if audio:
            voice_text = voice_to_text(audio['bytes'])
            if voice_text:
                st.success(f"🗣️: **'{voice_text}'**")
                updated = process_voice_command(
                    voice_text, 
                    district_names=list(district_display.values()), 
                    crop_names=list(all_crops_display.values()),
                    district_key="f_dist",
                    crop_key="f_crop"
                )
                if updated:
                    time.sleep(0.5)
                    st.rerun()

        f_district_bn = st.selectbox("জেলা নির্বাচন করুন", options=list(district_display.values()), key="f_dist")
        f_district = [k for k, v in district_display.items() if v == f_district_bn][0]
        
        f_crop_bn = st.selectbox("ফসল নির্বাচন করুন", options=list(all_crops_display.values()), key="f_crop")
        f_crop = [k for k, v in all_crops_display.items() if v == f_crop_bn][0]

        # Land Size
        land_amount = st.number_input("জমির পরিমাণ (শতাংশ/ডেসিমেল)", min_value=1.0, value=33.0, step=1.0)
    
    with c2:
        # Yield Estimation (Auto-fill based on data)
        avg_yield = 0
        crop_prod_data = prod_df[prod_df['Crop_Name'] == f_crop]
        if not crop_prod_data.empty:
            avg_yield = crop_prod_data['Yield_Quintals_per_Ha'].mean()
        
        # Convert Yield (Quintal/Hectare -> Kg/Decimal)
        # 1 Hectare = 247 Decimal
        # 1 Quintal = 100 kg
        # Yield (kg/dec) = (Yield_Q_Ha * 100) / 247
        default_yield_kg_dec = (avg_yield * 100) / 247 if avg_yield > 0 else 20.0
        
        expected_yield_per_dec = st.number_input("প্রত্যাশিত ফলন (কেজি/শতাংশ)", min_value=1.0, value=float(round(default_yield_kg_dec, 2)))
        
        # Price Estimation
        # Price Estimation
        # Get latest average price
        
        # Use AI Prediction if available
        pred_avg, _, _ = predict_next_30_days(f_district, f_crop, price_df)
        
        if pred_avg:
             default_price = pred_avg
        else:
             # Fallback to historical mean
             latest_price_date = price_df['Price_Date'].max()
             recent_prices = price_df[(price_df['Crop_Name'] == f_crop) & (price_df['Price_Date'] >= latest_price_date - datetime.timedelta(days=30))]
             default_price = recent_prices['Price_Tk_kg'].mean() if not recent_prices.empty else 20.0
        
        estimated_price = st.number_input("সম্ভাব্য বিক্রয় মূল্য (টাকা/কেজি)", min_value=1.0, value=float(round(default_price, 2)), help="AI দ্বারা পূর্বাভাসকৃত ৩০ দিনের গড় মূল্য (ডিফল্ট)")

    # 2. Generate Report
    if st.button("📄 রিপোর্ট তৈরি করুন", type="primary", use_container_width=True):
        total_production = land_amount * expected_yield_per_dec
        total_income = total_production * estimated_price
        
        # Cost Estimator (Rough Rule of Thumb: 40% of revenue is cost, usually higher but this is optimistic estimation for loan)
        # Better: Use static cost per decimal for simplicity
        estimated_cost = land_amount * 500 # Assuming 500 tk per decimal cost baseline
        net_profit = total_income - estimated_cost
        roi = (net_profit / estimated_cost) * 100 if estimated_cost > 0 else 0
        
        # Logic for Bank Loan Eligibility
        # If ROI > 30% and Profit > 20000, Good candidate
        loan_eligibility = "High" if roi > 30 and net_profit > 10000 else "Medium"
        if net_profit < 0: loan_eligibility = "None"
        
        st.divider()
        st.subheader("📋 এগ্রি-বিজনেস রিপোর্ট কার্ড")
        
        st.markdown(f"""
<div style="background-color: white; padding: 25px; border-radius: 12px; border: 1px solid #ddd; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
<div style="text-align: center; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; margin-bottom: 20px;">
<h2 style="color: #2E7D32; margin:0;">Agri-Business Projection</h2>
<p style="color: #666;">Generated by Agri-Smart BD AI</p>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
<div>
<p style="margin: 5px 0; color: #555;"><strong>জেলা:</strong> {f_district_bn}</p>
<p style="margin: 5px 0; color: #555;"><strong>ফসল:</strong> {f_crop_bn}</p>
<p style="margin: 5px 0; color: #555;"><strong>জমির পরিমাণ:</strong> {land_amount} শতাংশ</p>
<p style="margin: 5px 0; color: #555;"><strong>মোট উৎপাদন:</strong> {int(total_production)} কেজি</p>
</div>
<div>
<p style="margin: 5px 0; color: #555;"><strong>বাজার দর:</strong> ৳{estimated_price}/কেজি</p>
<p style="margin: 5px 0; color: #555;"><strong>আনুমানিক খরচ:</strong> ৳{int(estimated_cost)}</p>
</div>
</div>
<hr style="margin: 20px 0; border-top: 1px dashed #ccc;">
<div style="background-color: #f1f8e9; padding: 15px; border-radius: 8px; text-align: center;">
<h3 style="color: #1b5e20; margin: 0;">নিট মুনাফা (সম্ভাব্য)</h3>
<h1 style="color: #2e7d32; font-size: 2.5em; margin: 10px 0;">৳ {to_bengali_number(f'{int(net_profit)}')}/=</h1>
<p style="color: #33691e; font-weight: bold;">ROI: {roi:.1f}%</p>
</div>
<div style="margin-top: 20px; text-align: center;">
<span style="background-color: {'#4CAF50' if loan_eligibility == 'High' else '#FF9800'}; color: white; padding: 8px 15px; border-radius: 20px; font-weight: bold;">
ব্যাংক লোন যোগ্যতা: {loan_eligibility}
</span>
</div>
</div>
""", unsafe_allow_html=True)
        
        col_print, col_share = st.columns(2)
        with col_print:
            st.warning("🖨️ প্রিন্ট করতে 'Ctrl+P' চাপুন") 

# Footer
st.markdown("<br><hr><div style='text-align: center; color: #555;'>Agri-Smart BD | Built for AI Build-a-thon 2025</div>", unsafe_allow_html=True)