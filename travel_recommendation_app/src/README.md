# APAN5210FinalProjectgroup9

# Travel Recommendation App

Welcome to our Travel Recommendation App!  
This app uses a multi-agent system powered by LLMs to help users find top tourist attractions based on a location they want to explore. It supports multilingual output!

## Project Structure

src/
├── agents/
│   └── agents.py
├── shared_memory.py
├── streamlit_app.py
├── requirements.txt
└── README.md

## Features

- **Web Search Agent**: Searches for tourist attractions based on user location.
- **Recommender Agent**: Extracts attractions, gives short descriptions, and rates them.
- **Summarizer Agent**: Organizes recommendations into concise bullet points.
- **Translation Agent**: Translates recommendations into 10 languages.
- **Streamlit UI**: Interactive web interface.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone your-repo-link-here

cd your-repo-directory

```

### 2. Set Up Environment Variables
Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_key_here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run streamlit_app.py
```

## Sample Usage

- Input a location (e.g., "Thailand").
- Select a preferred language.
- Click **Get Recommendations**.
- Receive top-rated attractions with descriptions and ratings.

## Requirements

- Python 3.8+
- Required packages in `requirements.txt`:
  - `streamlit`
  - `openai`
  - `duckduckgo-search`
  - `python-dotenv`
  - `agno`


## Credits

Developed by group 9 as part of APAN 5210 — Spring 2025 Multi-Agent System Project.

