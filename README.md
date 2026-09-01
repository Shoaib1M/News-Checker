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

The checker now uses an evidence-first process:
*   **Claim extraction:** A multi-sentence submission is split into separately checkable claims.
*   **Evidence retrieval:** Search results locate candidate passages, including verification-focused queries.
*   **NLI verification:** A claim–evidence natural-language-inference model judges whether a passage entails, contradicts, or is neutral toward a claim.
*   **Conservative verdicts:** Only strong NLI judgments from classified primary, fact-checking, or reputable reporting sources can produce a verdict. Otherwise the result is **Insufficient evidence**.

The legacy LIAR MLP remains visible as an experimental US-political claim prior. It is not used to determine the final verdict, since production submissions do not include the historical speaker metadata used in its original evaluation.

## 📊 Model Performance

The custom **Binary Truth MLP** was trained on the challenging **LIAR dataset** (12,836 labelled political statements). To maximize real-world utility, the standard 6-class labels were collapsed into a binary "Fake-ish" vs "True-ish" classification.

### ⚠️ Legacy-model evaluation

The earlier **72.38%** result is a metadata-assisted LIAR test score: it uses the
speaker's historical truth counts, party, job, and context. It must not be
shown as the production model's performance.

The production-equivalent, statement-only evaluation is currently:

*   **Accuracy:** 62.35%
*   **Precision:** 63.18%
*   **Recall:** 79.55%
*   **F1 Score:** 0.7043
*   **Brier score:** 0.2262

This is deliberately treated as a legacy US-political prior, not an accuracy
claim for general news. The final application verdict comes from retrieved,
classified NLI evidence and abstains when that evidence is insufficient.

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

Before publishing an MLP accuracy number, run its production-equivalent
evaluation (the API only receives statement text):
```bash
cd ml-service
python evaluate_production_model.py
python -m unittest discover -s tests -v
```

The first evidence check downloads the NLI model configured by `NLI_MODEL`
(default: `cross-encoder/nli-deberta-v3-small`). The Docker deployment uses a
CPU-only PyTorch wheel and conservative thread limits so it fits on a 512 MB
Render instance. If the NLI model cannot be
loaded, the API deliberately returns **Insufficient evidence** rather than
falling back to keyword-based truth labels.

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
