from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelArguments

prompt_text = """
Based on the following:
Clinical: {{$clinical}}
Imaging: {{$imaging}}
Pathology: {{$pathology}}

Estimate overall risk (low/intermediate/high/aggressive/metastatic).
Match patient profile to possible clinical trials. Give rationale and eligibility criteria.
"""

def config():
    return PromptTemplateConfig(
        template=prompt_text,
        name="RiskTrialMatching",
        description="Stratifies risk and matches clinical trials"
    )

async def run(kernel, input_data):
    args = KernelArguments(**input_data)
    func = kernel.add_function(
        function_name="RiskTrialMatching",
        plugin_name="RiskAgent",
        prompt_template_config=config()
    )
    result = await kernel.invoke(function=func, arguments=args)
    return str(result)
