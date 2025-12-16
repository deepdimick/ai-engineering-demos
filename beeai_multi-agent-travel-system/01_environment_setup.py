import os

# Set up the watsonx.ai project ID
os.environ["WATSONX_PROJECT_ID"] = "input your watsonx project id"
os.environ["WATSONX_API_KEY"] = "your-watsonx-api-key"

# Set up the OpenAI API key and headers
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["OPENAI_API_HEADERS"] = "secret-header=1234"

print("Environment configured successfully!")