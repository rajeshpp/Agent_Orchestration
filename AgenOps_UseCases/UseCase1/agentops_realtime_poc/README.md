# AgentOps Real-time POC

This project demonstrates real-time monitoring and anomaly detection for stock data using AgentOps and OpenAI's GPT-4.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
AGENTOPS_API_KEY=your_agentops_api_key
```

## Running the Application

Run the application with:
```bash
python app.py
```

## Components

- `app.py`: Main application entry point
- `agent_logic.py`: Contains the RiskAgent class for analyzing stock data
- `data_feed.py`: Simulated stock data stream

## Features

- Real-time stock data monitoring
- Anomaly detection using GPT-4
- Event tracking with AgentOps
- Error handling and logging