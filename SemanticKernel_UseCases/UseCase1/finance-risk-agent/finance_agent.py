import os
import numpy as np
import asyncio
import semantic_kernel as sk

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelArguments
from dotenv import load_dotenv

# Load API keys
load_dotenv()

# Initialize Kernel
kernel = sk.Kernel()

chat_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    service_id="openai-chat",
    api_key=os.getenv("OPENAI_API_KEY")
)
kernel.add_service(chat_service)

# ---- Mock Market Data Fetcher ----
async def get_stock_prices(symbol: str):
    return np.random.normal(loc=100, scale=5, size=30)

# ---- Semantic Prompt ----
prompt_text = """
You are a financial advisor AI.
Analyze the following metrics:

Stock: {{$symbol}}
Volatility: {{$volatility}}

Classify risk as: Low / Medium / High  
Then provide a short reasoning and a recommendation (Buy / Hold / Sell)
with a professional tone.
"""

# Create the prompt template configuration
prompt_config = PromptTemplateConfig(
    template=prompt_text,
    name="EvaluateRisk",
    description="Evaluate stock risk from volatility"
)

# ✅ Create function from prompt using create_function_from_prompt
risk_function = kernel.add_function(
    function_name="EvaluateRisk",
    plugin_name="FinancePlugin",
    prompt_template_config=prompt_config
)

# ---- Main Pipeline ----
async def evaluate_stock(symbol: str):
    prices = await get_stock_prices(symbol)
    volatility = float(np.std(prices))
    
    # ✅ Use KernelArguments for passing parameters
    arguments = KernelArguments(
        symbol=symbol,
        volatility=f"{volatility:.2f}"
    )
    
    result = await kernel.invoke(
        function=risk_function,
        arguments=arguments
    )

    return {
        "symbol": symbol,
        "volatility": volatility,
        "advice": str(result)  # Convert result to string
    }

# CLI Run
if __name__ == "__main__":
    print("\n📈 Evaluating stock...\n")
    result = asyncio.run(evaluate_stock("AAPL"))
    print(f"\nSymbol: {result['symbol']}")
    print(f"Volatility: {result['volatility']:.2f}")
    print(f"\nAdvice:\n{result['advice']}")
