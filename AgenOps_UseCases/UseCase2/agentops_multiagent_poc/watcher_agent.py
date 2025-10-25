import agentops
import os
import traceback
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('WatcherAgent')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class WatcherAgent:
    def __init__(self):
        logger.info("Initializing WatcherAgent...")
        self.trace = agentops.start_trace(tags=["WatcherAgent", "Live"])
        logger.info("✅ WatcherAgent (Live) started")

    async def detect(self, data):
        """Detects significant movement in live price data."""
        try:
            logger.info("Starting detection for new data...")
            
            # Validate incoming data
            required_fields = ["symbol", "price", "price_history"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                logger.error(f"Missing required fields in data: {missing_fields}")
                return None
            
            symbol = data["symbol"]
            price = data["price"]
            price_history = data["price_history"]
            
            # Log incoming data structure
            logger.info(f"Processing data for {symbol}:")
            logger.info(f"  Current Price: ${price:.2f}")
            logger.info(f"  History Points: {len(price_history)}")
            logger.info(f"  Price History: {price_history}")
            logger.info(f"  Available Keys: {list(data.keys())}")
            
            # Skip analysis if we don't have enough history
            if len(price_history) < 2:
                logger.info(f"Not enough price history for {symbol}, need at least 2 points")
                return None
            
            # Calculate price change
            prev_price = price_history[-2] if len(price_history) > 1 else price
            pct_change = ((price - prev_price) / prev_price) * 100
            
            # Calculate statistics
            avg_price = sum(price_history) / len(price_history)
            std_dev = (sum((p - avg_price) ** 2 for p in price_history) / len(price_history)) ** 0.5
            z_score = (price - avg_price) / std_dev if std_dev != 0 else 0

            analysis_context = f"""
            Stock: {symbol}
            Current Price: ${price:.2f}
            Previous Price: ${prev_price:.2f}
            Percentage Change: {pct_change:.2f}%
            Average Price (last {len(price_history)} ticks): ${avg_price:.2f}
            Standard Deviation: ${std_dev:.2f}
            Z-Score (deviation from mean): {z_score:.2f}

            Should this be flagged for risk analysis? Consider:
            1. Absolute price movement (>3% is significant)
            2. Deviation from mean (z-score > 2 is unusual)
            3. Recent price trend

            Reply with either:
            'YES' if this needs risk analysis, with brief reason
            'NO' if this is normal movement
            """

            logger.info("Sending request to OpenAI for analysis...")
            try:
                logger.info(f"Sending analysis request to OpenAI for {symbol}")
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Using gpt-3.5-turbo instead of gpt-4 for better availability
                    messages=[
                        {"role": "system", "content": "You are a stock watcher AI agent focused on detecting unusual price movements."},
                        {"role": "user", "content": analysis_context}
                    ],
                    max_tokens=100
                )
                logger.info("Received response from OpenAI")
            except Exception as e:
                logger.error(f"OpenAI API call failed: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error traceback: {traceback.format_exc()}")
                raise

            result = response.choices[0].message.content.strip()
            agentops.tool(name="Watcher Analysis")({
                "symbol": symbol,
                "price": price,
                "pct_change": pct_change,
                "z_score": z_score,
                "analysis": result
            })

            if "yes" in result.lower():
                return {
                    "symbol": symbol,
                    "alert": True,
                    "context": analysis_context,
                    "current_price": price,
                    "pct_change": pct_change,
                    "z_score": z_score
                }
            return {"symbol": symbol, "alert": False}

        except Exception as e:
            agentops.tool(name="WatcherError")({
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return {"symbol": symbol, "alert": False}
