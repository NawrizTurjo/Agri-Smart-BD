# 🌾 Agri-Smart BD - AI-Powered Farm-to-Market Intelligence Platform

**Breaking the Middleman Syndicate | Empowering Farmers with Data-Driven Decisions**

A comprehensive agricultural intelligence dashboard for Bangladesh farmers, featuring AI-powered price forecasting with confidence intervals, smart market recommendations with transport cost analysis, and soil-based crop advisory. Built for the Trio Leveling for Bangladesh AI Build-a-thon 2025.

---

## 🎯 The Problem

In Bangladesh, farmers face a critical information gap:

- **Price Uncertainty:** Farmers don't know what price they'll get for their crops next week
- **Middlemen Exploitation:** Information asymmetry forces farmers into unfair deals with middleman syndicates
- **Lost Income:** Wrong market selection causes farmers to lose **15-20% of potential income**
- **Market Inefficiency:** Lack of real-time data prevents informed decision-making

**Agri-Smart BD** addresses these challenges by putting AI-powered market intelligence directly in the hands of farmers.

---

## 🚀 Core Features

### 📊 AI Price Forecasting with Uncertainty Analysis

- **Machine Learning Model:** Random Forest algorithm with 100 decision trees
- **30-Day Forecasts:** Future price predictions for various crops
- **Confidence Intervals:** Visual representation of price volatility and risk
- **Seasonality Detection:** Month and week-based pattern recognition
- **Historical Trends:** Interactive visualization of price movements
- **Smart Selling Recommendations:** Data-driven advice on optimal selling time

### 💰 Smart Market Finder & Net Profit Calculator

- **Transport Cost Integration:** Calculate actual profit after deducting transportation expenses
- **District-wise Comparison:** Real-time price comparison across all districts
- **Net Profit Ranking:** Identifies the most profitable markets for each crop
- **Interactive Visualizations:** Bar charts showing profit potential across regions
- **Syndicate-Free Pricing:** Empowers farmers to bypass middlemen with transparent data

### 🌱 Soil-Based Crop Advisor

- **Soil Health Dashboard:** Analysis of pH levels, nitrogen content, organic matter
- **Scientific Recommendations:** Top 5 crops suited for specific soil types and districts
- **Historical Yield Data:** Average production metrics for informed decisions
- **Reasoning Engine:** Explains why certain crops are recommended
- **Regional Adaptation:** District-specific crop performance insights

### 📲 SMS Alert System (Planned)

- **Feature Phone Support:** Designed for farmers without smartphones
- **Price Alerts:** Daily updates on crop prices
- **Market Recommendations:** SMS-based guidance
- **Current Status:** Mockup version (awaiting funding for SMS gateway integration)

## 🎨 User Interface

- **Bilingual Design:** Full Bengali (বাংলা) language support with English backend
- **Beautiful Gradient Theme:** Purple-themed professional dashboard
- **Responsive Layout:** Works seamlessly on desktop and mobile
- **Interactive Charts:** Plotly-powered visualizations
- **Accessible Design:** Clean interface suitable for farmers with varying tech literacy

---

## 📊 Data Strategy & Methodology

### Current Approach (Prototype)

Due to the lack of granular historical agricultural data for Bangladesh available during the hackathon timeframe, we employed a **strategic data engineering approach**:

- **Proxy Dataset:** Utilized agricultural data from **Rajasthan and West Bengal, India**
- **Rationale:** These regions share similar **agro-climatic conditions** with Bangladesh (soil types, monsoon patterns, crop varieties)
- **Adaptation:** Converted Indian district names, crop names, and market contexts to Bangladesh equivalents
- **Data Sources:**
  - Crop price data (district-wise, date-wise)
  - Crop production data (yield metrics)
  - Soil analysis data (pH, nutrients, organic matter)
  - Water usage patterns

### Future Data Collection Plan

To enhance **accuracy and real-world performance**, our roadmap includes:

1. **Extensive Market Surveys:** Conduct field surveys across major districts in Bangladesh
2. **Government Partnerships:** Collaborate with DAM (Department of Agricultural Marketing) for official data
3. **Farmer Input Programs:** Crowdsource real-time pricing data through mobile app
4. **IoT Integration:** Deploy soil sensors for live soil condition monitoring
5. **Historical Data Acquisition:** Partner with agricultural universities for research datasets

**Note:** This is a **functional prototype** demonstrating AI capabilities. With local Bangladesh data, prediction accuracy will significantly improve.

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/NawrizTurjo/Agri-Price-Pred-millionX.git
cd Agri-Price-Pred-millionX
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
Agri-Price-Pred-millionX/
├── app.py                          # Main Streamlit application with 3 modules
├── convert.py                      # Data conversion script (India → Bangladesh)
├── requirements.txt                # Python dependencies
├── bd_crop_price_data.csv          # Bangladesh crop price data (converted)
├── bd_crop_production_data.csv     # Bangladesh production data (converted)
├── bd_soil_analysis_data.csv       # Bangladesh soil data (converted)
├── bd_water_usage_data.csv         # Bangladesh water usage data (converted)
├── crop_price_data.csv             # Original Indian data (reference)
├── crop_production_data.csv        # Original Indian data (reference)
├── soil_analysis_data.csv          # Original Indian data (reference)
├── water_usage_data.csv            # Original Indian data (reference)
├── LICENSE                         # MIT License
└── README.md                       # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.13+**
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning (Random Forest)
- **Plotly** - Interactive visualizations

---

## 💡 Impact & Vision

### Immediate Impact

1. **Syndicate Breaking:** Real-time information prevents middlemen from exploiting farmers
2. **Profit Maximization:** Farmers can increase income by 15-20% through informed market selection
3. **Risk Management:** Confidence intervals help farmers understand price volatility
4. **Informed Decisions:** Soil-based recommendations improve crop selection and yield

### 10x Production Vision

Our goal is to help Bangladesh achieve **10x agricultural productivity growth** through:

- **Data-Driven Farming:** Every farmer has access to AI insights
- **Market Efficiency:** Transparent pricing reduces waste and maximizes value
- **Import Reduction:** Increased domestic production decreases dependency on imports
- **GDP Growth:** Agricultural sector contributes more significantly to national economy

### Social Impact

- **Economic Empowerment:** Farmers gain negotiating power with market knowledge
- **Rural Development:** Increased farmer income stimulates local economies
- **Food Security:** Better market efficiency ensures stable food supply
- **Digital Inclusion:** Brings AI benefits to underserved rural communities

---

## 🚀 Future Roadmap

### Phase 1: Mobile App Development (Q1-Q2 2026)

- **Offline-First Architecture:** Local database caching for areas with poor connectivity
- **Progressive Web App (PWA):** Works on both smartphones and feature phones
- **SMS Gateway Integration:** Real SMS alerts without internet dependency
- **Voice Commands:** Bengali voice input for illiterate farmers

### Phase 2: Data Enhancement (Q2-Q3 2026)

- **Field Market Surveys:** Collect real Bangladesh agricultural data
- **Government API Integration:** Connect with DAM pricing systems
- **Farmer Crowdsourcing:** Community-contributed price data
- **IoT Sensors:** Deploy soil and weather monitoring devices

### Phase 3: Advanced Features (Q3-Q4 2026)

- **Weather Integration:** Combine weather forecasts with price predictions
- **Crop Disease Detection:** AI-powered image recognition for plant health
- **Supply Chain Tracking:** Farm-to-consumer transparency
- **Cooperative Formation:** Tools for farmers to organize and negotiate collectively

### Phase 4: Regional Expansion (2027+)

- **South Asian Rollout:** Expand to Nepal, Bhutan, Sri Lanka
- **Multi-language Support:** Additional regional languages
- **Blockchain Integration:** Transparent supply chain records
- **Microfinance Partnerships:** Credit access based on crop predictions

---

## 👥 Team

**Team Trio Leveling**  
_Trio Leveling for Bangladesh AI Build-a-thon 2025_

We are a team of passionate developers and data scientists committed to using technology for social impact.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Commercial Use:** Free and open-source for agricultural development purposes.

---

## 🙏 Acknowledgments

- **Trio Leveling Bangladesh** for organizing the AI Build-a-thon 2025
- **Data Sources:** Indian agricultural datasets (Rajasthan & West Bengal) used as proxy
- **Inspiration:** The hardworking farmers of Bangladesh who deserve better market access
- **Open Source Community:** Streamlit, Scikit-learn, and Plotly teams

---

## 🤝 Contributing

We welcome contributions! If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Contact & Support

**For inquiries, partnerships, or data contributions:**

- 📧 Email: [Contact through GitHub Issues]
- 🌐 Live Demo: [https://agribangla.streamlit.app/](https://agribangla.streamlit.app/)
- 💻 Repository: [GitHub - MXB2026-Dhaka-Trio-Leveling-AgriSmartBD](https://github.com/Your-Tribal-Chief/MXB2026-Dhaka-Trio-Leveling-AgriSmartBD/)

---

<div align="center">

**Built with ❤️ for Bangladesh Farmers**

_"Technology should empower those who feed nations"_

</div>and seasonality detection
