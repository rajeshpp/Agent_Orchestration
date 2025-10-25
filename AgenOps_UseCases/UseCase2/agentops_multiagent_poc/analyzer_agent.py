import agentops
import os
import traceback
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AnalyzerAgent')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AnalyzerAgent:
    def __init__(self):
        logger.info("Initializing AnalyzerAgent...")
        self.trace = agentops.start_trace(tags=["AnalyzerAgent"])
        logger.info("✅ AnalyzerAgent trace started")

    async def analyze(self, context):
        """Analyze flagged anomalies and assess risk severity."""
        logger.info("Starting analysis of flagged anomaly...")
        
        prompt = f"""
        Given this detailed stock analysis:
        {context}

        Provide a structured risk assessment in the following format:

        RISK LEVEL: [Low/Medium/High]
        CAUSE: [Brief description of potential causes]
        IMPACT: [Market implications]
        ACTION: [Recommended monitoring/action steps]

        Keep each section concise but informative. Focus on actionable insights.
        """
        logger.info("Analysis prompt prepared")

        try:
            logger.info("Sending request to OpenAI for risk analysis...")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using gpt-3.5-turbo instead of gpt-4 for better availability
                messages=[
                    {"role": "system", "content": "You are a risk analysis agent."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=80,
            )
            logger.info("Received response from OpenAI")

            result = response.choices[0].message.content.strip()
            agentops.tool(name="Analyzer Assessment")({
                "input": context,
                "output": result,
                "tags": ["analyzer"]
            })

            return f"🔍 {result}"

        except Exception as e:
            agentops.tool(name="AnalyzerError")({
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return "Analysis failed."
