from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelArguments

prompt_text = """
Patient summary:
{{$all_findings}}

Based on major oncology guidelines, recommend a personalized cancer treatment plan.
Consider: surgery, radiation, chemotherapy, immunotherapy, targeted therapy, or observation.
Simulate possible outcomes and risks. Explain your rationale.
"""

def config():
    return PromptTemplateConfig(
        template=prompt_text,
        name="TreatmentPlanning",
        description="Guideline-based treatment suggestions"
    )

async def run(kernel, input_data):
    args = KernelArguments(**input_data)
    func = kernel.add_function(
        function_name="TreatmentPlanning",
        plugin_name="TreatmentAgent",
        prompt_template_config=config()
    )
    result = await kernel.invoke(function=func, arguments=args)
    return str(result)
