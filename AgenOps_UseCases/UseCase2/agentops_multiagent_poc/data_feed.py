import asyncio
import random
import datetime
from collections import defaultdict, deque

STOCKS = ["AAPL", "GOOG", "TSLA", "AMZN", "META"]
MAX_HISTORY = 10  # Keep last 10 data points for each stock

class StockDataGenerator:
    def __init__(self):
        self.price_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
        # Initialize with some historical data
        for symbol in STOCKS:
            base_price = random.uniform(100, 500)
            for _ in range(MAX_HISTORY):
                # Add some random variation to create history
                price = round(base_price * (1 + random.uniform(-0.02, 0.02)), 2)
                self.price_history[symbol].append(price)

    def get_next_price(self, symbol, allow_anomaly=True):
        """Generate next price with possible anomalies"""
        last_price = self.price_history[symbol][-1]
        if allow_anomaly and random.random() < 0.1:  # 10% chance of anomaly
            # Generate significant price movement (±5-15%)
            change = random.uniform(0.05, 0.15) * (-1 if random.random() < 0.5 else 1)
            new_price = round(last_price * (1 + change), 2)
        else:
            # Normal price movement (±0.5-2%)
            change = random.uniform(0.005, 0.02) * (-1 if random.random() < 0.5 else 1)
            new_price = round(last_price * (1 + change), 2)
        
        self.price_history[symbol].append(new_price)
        return new_price

    async def stream_stock_data(self):
        """Simulated live stock feed with price history."""
        while True:
            symbol = random.choice(STOCKS)
            price = self.get_next_price(symbol)
            data = {
                "symbol": symbol,
                "price": price,
                "timestamp": datetime.datetime.now().isoformat(),
                "price_history": list(self.price_history[symbol])
            }
            yield data
            await asyncio.sleep(2)

async def stream_stock_data():
    """Entry point for stock data streaming"""
    generator = StockDataGenerator()
    async for data in generator.stream_stock_data():
        yield data
