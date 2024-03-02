import os

import google.generativeai as genai

generation_config = genai.types.GenerationConfig(
    temperature=1,
    top_p=1,
    top_k=1,
    max_output_tokens=4000,
)


class MyModel:

  def __init__(self):
    pass

  def run_gemini(self, user_input):

    INSTRUCTION = f"""
        As a helpful assistant, your task is to respond to user queries. Below is the user input enclosed within triple backticks:

        User input: ```{user_input}```

        If the requested information is available in your knowledge base, respond with your response. Ensure to use escape characters in your response to avoid syntax errors while parsing. For example:
        1. Use escape characters for triple double quotes: \"\"\"
        2. Double quotes: \"
        3. New line: \\n
        

        If the requested information is not available in your knowledge base, respond with "not found"
    """

    genai.configure(api_key=os.environ['GEMINI_API'])

    model_gem = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                      generation_config=generation_config)

    response = model_gem.generate_content(INSTRUCTION)

    final = {"answer": response.text}
    return final