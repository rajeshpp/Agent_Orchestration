from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelArguments

prompt_text = """
Pathology & Genomics:
Biopsy Findings: {{$findings}}
Molecular Markers: {{$markers}}

Summarize cancer type, grade, prognosis, and implications for therapy.
"""

def config():
    return PromptTemplateConfig(
        template=prompt_text,
        name="PathologyGenomics",
        description="Integrate pathology and genomics data"
    )

async def run(kernel, input_data):
    args = KernelArguments(**input_data)
    func = kernel.add_function(
        function_name="PathologyGenomics",
        plugin_name="PathologyAgent",
        prompt_template_config=config()
    )
    result = await kernel.invoke(function=func, arguments=args)
    return str(result)
