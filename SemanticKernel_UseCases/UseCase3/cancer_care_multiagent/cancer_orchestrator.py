import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from dotenv import load_dotenv
import asyncio

from agents import clinical_intake, imaging_analysis, pathology_genomics, risk_trial_matching, treatment_planning

load_dotenv()
kernel = Kernel()
chat = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    service_id="openai-chat",
    api_key=os.getenv("OPENAI_API_KEY")
)
kernel.add_service(chat)

async def cancer_care_pipeline(patient):
    steps = {}

    # 1. Clinical Intake Agent
    steps["clinical"] = await clinical_intake.run(kernel, patient["clinical"])

    # 2. Imaging Analysis Agent
    steps["imaging"] = await imaging_analysis.run(kernel, patient["imaging"])

    # 3. Pathology/Genomics Agent
    steps["pathology"] = await pathology_genomics.run(kernel, patient["pathology"])

    # 4. Risk/Trial Agent
    risk_input = {
        "clinical": steps["clinical"],
        "imaging": steps["imaging"],
        "pathology": steps["pathology"]
    }
    steps["risk"] = await risk_trial_matching.run(kernel, risk_input)

    # 5. Treatment Planning Agent
    treatment_input = {
        "all_findings": '\n---\n'.join([
            steps["clinical"], steps["imaging"], steps["pathology"], steps["risk"]
        ])
    }
    steps["treatment"] = await treatment_planning.run(kernel, treatment_input)

    return steps

# AIOHTTP Server
from aiohttp import web

async def handle_dashboard(request):
    with open('dashboard.html', 'r', encoding='utf-8') as f:
        return web.Response(text=f.read(), content_type='text/html')

async def handle_orchestrate(request):
    try:
        data = await request.json()
        results = await cancer_care_pipeline(data)
        return web.json_response(results)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

app = web.Application()
app.router.add_get('/', handle_dashboard)
app.router.add_post('/api/run', handle_orchestrate)

if __name__ == "__main__":
    print("Cancer multi-agent server at http://localhost:8080")
    web.run_app(app, port=8080)
