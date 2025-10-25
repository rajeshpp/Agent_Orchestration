import os
import numpy as np
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelArguments
from dotenv import load_dotenv
from aiohttp import web
import json

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

# ---- Mock Patient Vital Signs Simulator ----
# In production, replace with real EHR/EMR API integration
def get_patient_vitals(patient_id: str):
    """Simulate patient vital signs with realistic variations"""
    base_vitals = {
        "heart_rate": np.random.normal(loc=75, scale=10),  # Normal: 60-100 bpm
        "blood_pressure_systolic": np.random.normal(loc=120, scale=15),  # Normal: 90-120
        "blood_pressure_diastolic": np.random.normal(loc=80, scale=10),  # Normal: 60-80
        "temperature": np.random.normal(loc=98.6, scale=0.8),  # Normal: 97.7-99.1°F
        "respiratory_rate": np.random.normal(loc=16, scale=3),  # Normal: 12-18 breaths/min
        "oxygen_saturation": np.random.normal(loc=98, scale=2),  # Normal: 95-100%
    }
    return {k: round(v, 1) for k, v in base_vitals.items()}

# ---- Semantic Prompt for Health Risk Assessment ----
prompt_text = """
You are an AI-powered healthcare assistant helping doctors assess patient risk levels.

Analyze the following patient vital signs:

Patient ID: {{$patient_id}}
Heart Rate: {{$heart_rate}} bpm (Normal: 60-100)
Blood Pressure: {{$bp_systolic}}/{{$bp_diastolic}} mmHg (Normal: 90-120/60-80)
Temperature: {{$temperature}}°F (Normal: 97.7-99.1)
Respiratory Rate: {{$respiratory_rate}} breaths/min (Normal: 12-18)
Oxygen Saturation: {{$oxygen_saturation}}% (Normal: 95-100)

Based on these vitals:
1. Classify overall health risk as: LOW / MODERATE / HIGH / CRITICAL
2. Identify any abnormal vital signs
3. Provide clinical recommendations
4. Suggest immediate actions if needed

Format your response professionally for healthcare providers.
"""

prompt_config = PromptTemplateConfig(
    template=prompt_text,
    name="AssessHealthRisk",
    description="Assess patient health risk from vital signs"
)

# Create the health risk assessment function
risk_function = kernel.add_function(
    function_name="AssessHealthRisk",
    plugin_name="HealthcarePlugin",
    prompt_template_config=prompt_config
)

# ---- Main Assessment Pipeline ----
async def assess_patient_risk(patient_id: str):
    vitals = get_patient_vitals(patient_id)
    
    arguments = KernelArguments(
        patient_id=patient_id,
        heart_rate=str(vitals["heart_rate"]),
        bp_systolic=str(vitals["blood_pressure_systolic"]),
        bp_diastolic=str(vitals["blood_pressure_diastolic"]),
        temperature=str(vitals["temperature"]),
        respiratory_rate=str(vitals["respiratory_rate"]),
        oxygen_saturation=str(vitals["oxygen_saturation"])
    )
    
    result = await kernel.invoke(
        function=risk_function,
        arguments=arguments
    )

    return {
        "patient_id": patient_id,
        "vitals": vitals,
        "assessment": str(result)
    }

# ---- Web API for UI Integration ----
async def handle_assessment(request):
    """API endpoint for health risk assessment"""
    try:
        data = await request.json()
        patient_id = data.get('patient_id', 'P001')
        
        result = await assess_patient_risk(patient_id)
        
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def handle_index(request):
    """Serve the UI dashboard"""
    with open('health_dashboard.html', 'r', encoding='utf-8') as f:
        return web.Response(text=f.read(), content_type='text/html')


# CLI Run
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        # Run as web server
        app = web.Application()
        app.router.add_get('/', handle_index)
        app.router.add_post('/api/assess', handle_assessment)
        
        print("\n🏥 Healthcare Risk Assessment Server Starting...")
        print("📊 Dashboard: http://localhost:8080")
        print("🔌 API: http://localhost:8080/api/assess\n")
        
        web.run_app(app, host='localhost', port=8080)
    else:
        # Run CLI assessment
        print("\n🏥 Assessing patient health risk...\n")
        result = asyncio.run(assess_patient_risk("P12345"))
        
        print(f"Patient ID: {result['patient_id']}")
        print("\n📊 Vital Signs:")
        for key, value in result['vitals'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print(f"\n🤖 AI Assessment:\n{result['assessment']}")
