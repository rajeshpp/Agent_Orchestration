import agentops
from openai import OpenAI
import os
import traceback
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RiskAgent:
    def __init__(self):
        # one trace per agent lifetime
        self.trace = agentops.start_trace(tags=["realtime", "finance"])
        print("AgentOps trace started")

    async def process(self, data):
        """
        Evaluate incoming stock data and detect anomalies.
        """
        # Calculate basic statistics from history
        history = data['price_history'][:-1]  # All except current price
        current_price = data['price']
        avg_price = sum(history) / len(history)
        price_std = (sum((p - avg_price) ** 2 for p in history) / len(history)) ** 0.5
        
        # Calculate percentage change from previous price
        prev_price = history[-1]
        pct_change = ((current_price - prev_price) / prev_price) * 100

        prompt = f"""
        Analyze this stock tick for {data['symbol']}:
        - Current Price: ${current_price:.2f}
        - Previous Price: ${prev_price:.2f}
        - Percentage Change: {pct_change:.2f}%
        - Average Price (last {len(history)} ticks): ${avg_price:.2f}
        - Price Standard Deviation: ${price_std:.2f}

        Historical context:
        Last {len(history)} prices: {', '.join(f'${p:.2f}' for p in history)}

        Detect if there's any anomaly (sudden drop/spike) based on:
        1. Deviation from moving average
        2. Percentage change from previous price
        3. Comparison with historical volatility

        If no significant anomaly, reply exactly with 'No issue.'
        If there is an anomaly, explain the specific pattern detected.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            result = response.choices[0].message.content.strip()

            # Log both the analysis and the data
            agentops.tool(name="Stock Data Analysis")({
                "symbol": data['symbol'],
                "current_price": current_price,
                "pct_change": pct_change,
                "avg_price": avg_price,
                "price_std": price_std,
                "analysis": result
            })

            if "No issue" not in result:
                return f"{data['symbol']} Risk Alert → {result}"

        except Exception as e:
            agentops.tool(name="Processing Error")({"error": str(e)})
            return None
