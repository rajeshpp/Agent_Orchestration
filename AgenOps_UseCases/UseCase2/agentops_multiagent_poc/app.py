import asyncio
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import agentops
from watcher_agent import WatcherAgent
from analyzer_agent import AnalyzerAgent
from data_feed_live import stream_stock_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('MainApp')

# Ensure .env file is loaded first
env_path = Path(__file__).parent / '.env'
if not env_path.exists():
    print("Error: .env file not found! Please create one with required API keys")
    sys.exit(1)

# Load environment variables
load_dotenv(dotenv_path=env_path)

# Verify required environment variables
required_vars = ['OPENAI_API_KEY', 'AGENTOPS_API_KEY', 'FINNHUB_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please add them to your .env file.")
    sys.exit(1)

# Initialize AgentOps
agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))

async def main():
    logger.info("\n" + "=" * 80)
    logger.info("🚀 AgentOps Multi-Agent Live POC Started...")
    logger.info("=" * 80 + "\n")
    
    logger.info("Initializing agents...")
    watcher = WatcherAgent()
    analyzer = AnalyzerAgent()
    logger.info("Agents initialized successfully")

    try:
        async for data in stream_stock_data(interval=15):  # fetch every 15 sec
            # Show the incoming data
            print(f"\n📊 Processing {data['symbol']} - ${data['price']:.2f}")
            if len(data['price_history']) > 1:
                pct_change = ((data['price'] - data['price_history'][-2]) / data['price_history'][-2]) * 100
                print(f"   Change: {pct_change:+.2f}% | Volume: {data.get('volume', 'N/A')}")
            
            watch_result = await watcher.detect(data)
            
            if not watch_result:
                print("   ⏳ No significant movement detected")
                continue

            if watch_result.get("alert"):
                print("\n" + "🚨 " * 20)
                print(f"⚠️  ALERT: Unusual movement detected for {data['symbol']}")
                print(f"   Current Price: ${data['price']:.2f}")
                if 'pct_change' in watch_result:
                    print(f"   Change: {watch_result['pct_change']:+.2f}%")
                if 'z_score' in watch_result:
                    print(f"   Z-Score: {watch_result['z_score']:.2f}")
                
                analysis = await analyzer.analyze(watch_result["context"])
                print("\n🔍 Analysis:")
                print(analysis)
                print("🚨 " * 20 + "\n")
            else:
                print("   ✅ Normal market behavior")
            
            # Add a visual separator between stocks
            print("-" * 40)

    except KeyboardInterrupt:
        print("\n\n⚡ Stopping the stock monitoring system...")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    finally:
        print("\n" + "=" * 80)
        print("🏁 AgentOps Multi-Agent Live POC Completed")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
