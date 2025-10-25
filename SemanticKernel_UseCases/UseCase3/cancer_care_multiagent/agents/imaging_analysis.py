from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelArguments

prompt_text = """
Imaging Findings:
Modality: {{$type}}
Findings: {{$findings}}
Tumor Size: {{$size}}
Location: {{$location}}

Analyze for malignancy, staging, and any urgent concerns.
"""

def config():
    return PromptTemplateConfig(
        template=prompt_text,
        name="ImagingAnalysis",
        description="Analyze cancer imaging"
    )

async def run(kernel, input_data):
    args = KernelArguments(**input_data)
    func = kernel.add_function(
        function_name="ImagingAnalysis",
        plugin_name="ImagingAgent",
        prompt_template_config=config()
    )
    result = await kernel.invoke(function=func, arguments=args)
    return str(result)
