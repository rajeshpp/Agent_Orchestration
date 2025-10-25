from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelArguments

prompt_text = """
Patient Clinical Intake:
Name: {{$name}}
Age: {{$age}}
Sex: {{$sex}}
Symptoms: {{$symptoms}}
Family History: {{$family_history}}
Prior Cancer: {{$prior_cancer}}

Summarize patient risk factors and relevant clinical details for oncology.
"""

def config():
    return PromptTemplateConfig(
        template=prompt_text,
        name="ClinicalIntake",
        description="Summarize patient clinical intake"
    )

async def run(kernel, input_data):
    args = KernelArguments(**input_data)
    func = kernel.add_function(
        function_name="ClinicalIntake",
        plugin_name="IntakeAgent",
        prompt_template_config=config()
    )
    result = await kernel.invoke(function=func, arguments=args)
    return str(result)
