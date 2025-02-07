import yaml

def load_yaml_prompts(model, language):
    with open("src/services/llm/prompts.yaml", "r") as file:
        prompts = yaml.safe_load(file)

    prompt = prompts[model]['system_prompt'].format(language=language)
    return prompt
    
# print(load_yaml_prompts('llama3.2:1b', 'en'))


