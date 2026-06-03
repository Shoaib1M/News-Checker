# NewsChecker: AI-Powered Fact-Checking System

NewsChecker is a full-stack, AI-powered fact-checking application that evaluates the credibility of political statements and news claims. 

Unlike typical wrappers around third-party APIs (like OpenAI), this project features a **custom-built Machine Learning pipeline constructed from scratch** using NumPy, paired with real-time web scraping for evidence corroboration.

## 🚀 Features

*   **Custom Neural Network:** A Multi-Layer Perceptron (MLP) built from scratch without PyTorch or TensorFlow, trained on the LIAR dataset.
*   **Custom NLP Vectorization:** A bespoke TF-IDF implementation designed specifically for text classification.
*   **Real-time Evidence Gathering:** Automatically scrapes DuckDuckGo, Guardian, GNews, and NewsAPI to cross-reference claims against live web data.
*   **Stance Detection:** Analyzes scraped articles to determine if they support or contradict the claim using keyword overlap, directional agreement, and negation detection.
*   **Microservice Architecture:** A Node.js/Express backend handles user data, while a dedicated Python/FastAPI service handles the heavy lifting of ML inference and web scraping.
*   **Secure Authentication:** Integrates Google OAuth 2.0 with JSON Web Tokens (JWT) for secure user sessions.
*   **Interactive Data Science Visualizations:** Custom-built React components that render Confusion Matrices and ROC curves dynamically using raw SVG and CSS.

## 🏗️ Architecture

The application is split into three distinct layers:

1.  **Frontend (`client/`)**
    *   **Tech Stack:** React, Vite, Vanilla CSS.
    *   **Role:** Provides a responsive, animated user interface. It manages state using React hooks and visualizes the ML model's confidence scores alongside the scraped evidence.

2.  **Backend API (`server/`)**
    *   **Tech Stack:** Node.js, Express.js, MongoDB, Mongoose.
    *   **Role:** Acts as the gateway for the frontend. Handles Google OAuth login, issues JWTs, stores user fact-check history in MongoDB, and proxies prediction requests to the Python ML Service.

3.  **Machine Learning Service (`ml-service/`)**
    *   **Tech Stack:** Python, FastAPI, NumPy, Pandas, BeautifulSoup4.
    *   **Role:** Exposes the `/predict` endpoint. It vectorizes the incoming text, runs it through the trained Neural Network to get an initial confidence score, scrapes the web for supporting/contradicting evidence, calculates the net stance, and returns a blended final score.

## 🧮 How the Scoring Works

The final credibility score (0-100) is a weighted combination of three signals:
*   **40% ML Confidence:** The raw sigmoid probability output from the custom-built neural network.
*   **35% Evidence Similarity:** The cosine similarity between the user's claim and the text of the scraped articles.
*   **25% Evidence Stance:** A calculated score representing whether the overall tone of the scraped articles supports or contradicts the claim.

## 📊 Model Performance

The custom **Binary Truth MLP** was trained on the challenging **LIAR dataset** (12,836 labelled political statements). To maximize real-world utility, the standard 6-class labels were collapsed into a binary "Fake-ish" vs "True-ish" classification.

### 🏆 Outperforming the Baseline
A standard Logistic Regression model on this dataset achieved **56.35%** accuracy. 
Our custom-built Neural Network with engineered features achieved a **~28% relative performance increase**:

*   **Accuracy:** 72.38% *(+16.03% over baseline)*
*   **Precision:** 78.09% 
*   **Recall:** 70.87%
*   **F1 Score:** 0.743
*   **AUC:** 0.7794

### 🧠 Architectural Advantages
*   **Dynamic Thresholding:** Rather than using a naive 0.5 decision boundary, the sigmoid threshold is dynamically tuned on the validation set (currently optimized at **0.53**) to maximize accuracy.
*   **Engineered Features:** It doesn't just look at text; the vector space is augmented with the speaker's historical truth-counts, party affiliation, and job title.
*   **Production Blending:** The 72% base accuracy of the isolated ML model is heavily augmented in production by the live web evidence similarity and stance detection, pushing the final system accuracy significantly higher.

## 💻 Local Development Setup

To run this project locally, you will need to start all three services.

### Prerequisites
*   Node.js (v18+)
*   Python (3.9+)
*   MongoDB (Running locally or via MongoDB Atlas)

### 1. Start the Machine Learning Service
```bash
cd ml-service
# It is recommended to create a virtual environment first
pip install -r requirements.txt
python main.py
```
*The FastAPI server will start on `http://localhost:8000`*

### 2. Start the Node.js Server
```bash
cd server
npm install
npm run dev
```
*The Express server will start on `http://localhost:5000`*

### 3. Start the React Frontend
```bash
cd client
npm install
npm run dev
```
*The Vite development server will start on `http://localhost:5173`*

## 📚 Educational Nature

This entire codebase is heavily documented for educational purposes. Throughout the files, you will find detailed `FILE PURPOSE`, `FLOW`, and `WHY THIS EXISTS` headers. 

If you are learning about Machine Learning algorithms, React state management, or Full-Stack microservice architectures, this repository is designed to be read like a textbook.

## 📄 License
This project is open-source and available under the MIT License.
