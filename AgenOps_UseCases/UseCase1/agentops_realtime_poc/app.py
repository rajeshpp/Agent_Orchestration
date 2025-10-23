import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from data_feed import stream_stock_data
from agent_logic import RiskAgent
import agentops

# Ensure .env file is loaded first
env_path = Path(__file__).parent / '.env'
if not env_path.exists():
    print("Error: .env file not found! Please create one with OPENAI_API_KEY and AGENTOPS_API_KEY")
    sys.exit(1)

# Load environment variables from .env
load_dotenv(dotenv_path=env_path)

# Verify required environment variables
required_vars = ['OPENAI_API_KEY', 'AGENTOPS_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please add them to your .env file.")
    sys.exit(1)

# Initialize AgentOps with API key from .env
agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))

async def main():
    print("AgentOps Realtime POC Started...")
    agent = RiskAgent()
    async for data in stream_stock_data():
        response = await agent.process(data)
        if response:
            agentops.tool(name="Risk Assessment")({"input": data, "output": response})
            print("*" * 50)
            print(f"{response}")

if __name__ == "__main__":
    asyncio.run(main())
