# Import necessary libraries and modules
import asyncio
import logging
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage, SystemMessage

# Define the basic chat example function
async def basic_chat_example():
    # Create a ChatModel instance with the specified name and parameters
    llm = ChatModel.from_name("watsonx:ibm/granite-3-3-8b-instruct", ChatModelParameters(temperature=0))
    
    # Define the system and user messages
    messages = [
        SystemMessage(content="You are a helpful business assistant that answers questions about company operations, policies, and processes with clarity and professionalism."),
        UserMessage(content="What is our company's remote work policy and how do I request time off?")
    ]
    
    # Create the response using the ChatModel instance
    response = await llm.create(messages=messages)
    
    print("User: What is our company's remote work policy and how do I request time off?")
    print(f"Assistant: {response.get_text_content()}")
    
    return response

    # Run the main function to execute the basic chat example
async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL) # Suppress warnings from asyncio
    response = await basic_chat_example()
if __name__ == "__main__":
    asyncio.run(main())