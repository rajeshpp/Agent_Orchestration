import aiohttp
import asyncio
import os
import logging
from collections import defaultdict, deque
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('StockDataFeed')

# Load environment variables
load_dotenv()
API_KEY = os.getenv("FINNHUB_API_KEY")
if not API_KEY:
    raise ValueError("FINNHUB_API_KEY not found in environment variables")

STOCKS = ["AAPL", "GOOG", "TSLA", "AMZN", "MSFT"]
MAX_HISTORY = 10  # Keep last 10 data points for each stock

class LiveStockDataManager:
    def __init__(self):
        logger.info("Initializing LiveStockDataManager...")
        self.price_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
        self.session = None
        logger.info("LiveStockDataManager initialized")

    async def initialize_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_stock_price(self, symbol):
        """Fetch live stock price from Finnhub API."""
        logger.info(f"Fetching price for {symbol}...")
        if not self.session:
            logger.info("Initializing new session...")
            await self.initialize_session()

        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}"
        headers = {'X-Finnhub-Token': API_KEY}
        logger.debug(f"Requesting URL: {url}")
        
        async with self.session.get(url, headers=headers) as resp:
            if resp.status != 200:
                logger.error(f"Error fetching {symbol}: HTTP {resp.status}")
                return None
            
            try:
                data = await resp.json()
                logger.debug(f"Raw response for {symbol}: {data}")
                
                if 'c' not in data:  # Current price
                    logger.error(f"No quote data received for {symbol}. Response: {data}")
                    return None

                price = float(data['c'])  # Current price
                if price <= 0:
                    logger.error(f"Invalid price {price} received for {symbol}")
                    return None
                    
                # Update price history
                self.price_history[symbol].append(price)
                
                # Calculate percentage change
                prev_close = data.get('pc', price)  # Previous close price
                change_percent = ((price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                
                return {
                    "symbol": symbol,
                    "price": price,
                    "volume": int(data.get('v', 0)),  # Volume
                    "change_percent": f"{change_percent:.2f}%",
                    "price_history": list(self.price_history[symbol])
                }
            except (KeyError, ValueError, TypeError):
                return None

    async def __aenter__(self):
        await self.initialize_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()

async def stream_stock_data(interval=10):
    """Continuously stream live stock data with price history."""
    logger.info(f"Starting stock data stream with {len(STOCKS)} symbols, interval: {interval}s")
    async with LiveStockDataManager() as manager:
        while True:
            logger.info("Fetching new batch of stock data...")
            for symbol in STOCKS:
                stock_data = await manager.fetch_stock_price(symbol)
                if stock_data:
                    logger.info(f"Yielding data for {symbol}:")
                    logger.info(f"  Price: ${stock_data['price']:.2f}")
                    logger.info(f"  History: {stock_data['price_history']}")
                    logger.info(f"  Volume: {stock_data.get('volume', 'N/A')}")
                    logger.info(f"  Change %: {stock_data.get('change_percent', 'N/A')}")
                    yield stock_data
                else:
                    logger.warning(f"No data received for {symbol}")
            logger.info(f"Sleeping for {interval} seconds...")
            await asyncio.sleep(interval)
