# Import necessary libraries
import logging
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

# Establish logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the LlamaVisionService class
class LlamaVisionService:
    """
    Allows for interaction with the Llama 3.2 Vision Instruct model.
    """
    
    # Initialize the LlamaVisionService with necessary parameters
    def __init__(self, model_id, project_id, region="us-south", 
                 temperature=0.2, top_p=0.6, api_key=None, max_tokens=2000):
        
        # Initialize credentials and client
        credentials = Credentials(
            url=f"https://{region}.ml.cloud.ibm.com",
            api_key=api_key
        )
        self.client = APIClient(credentials)
        
        # Define the text chat parameters
        params = TextChatParameters(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        # Initialize the model with the provided parameters
        self.model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
            params=params
        )
    
    # Define the generate_response method to send requests to the LLM
    def generate_response(self, encoded_image, prompt):
       
        try:
            logger.info("Sending request to LLM with prompt length: %d", len(prompt))
            
            # Create the message structure
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + encoded_image,
                            }
                        }
                    ]
                }
            ]
            
            # Send the message to the model
            response = self.model.chat(messages=messages)
            
            # Extract the content from the response
            content = response['choices'][0]['message']['content']
            
            logger.info("Received response with length: %d", len(content))
            
            # Verify the response length
            if len(content) >= 7900:  # Close to common model limits
                logger.warning("Response may be truncated (length: %d)", len(content))
            
            return content
        
        # Handle specific exceptions    
        except Exception as e:
            logger.error("Error generating response: %s", str(e))
            return f"Error generating response: {e}"
    
    # Define the generate_fashion_response method to handle fashion analysis
    def generate_fashion_response(self, user_image_base64, matched_row, all_items, 
                                 similarity_score, threshold=0.8):
         # Generate a list of item descriptions with name, price, and link
        items_list = []
        for _, row in all_items.iterrows():
            item_str = f"{row['Item Name']} (${row['Price']}): {row['Link']}"
            items_list.append(item_str)
        
        # Join the items list into a single string with newline separation
        items_description = "\n".join([f"- {item}" for item in items_list])
        
        # Determine the prompt based on similarity score
        if similarity_score >= threshold:
            # Prompt for highly similar items
            assistant_prompt = (
                f"You're conducting a professional retail catalog analysis. "
                f"This image shows standard clothing items available in department stores. "
                f"Focus exclusively on professional fashion analysis for a clothing retailer. "
                f"ITEM DETAILS (always include this section in your response):\n{items_description}\n\n"
                "Please:\n"
                "1. Identify and describe the clothing items objectively (colors, patterns, materials)\n"
                "2. Categorize the overall style (business, casual, etc.)\n"
                "3. Include the ITEM DETAILS section at the end\n\n"
                "This is for a professional retail catalog. Use formal, clinical language."
                    )
        else:
            # Alternative prompt for less similar items
            assistant_prompt = (
                f"You're conducting a professional retail catalog analysis. "
                f"This image shows standard clothing items available in department stores. "
                f"Focus exclusively on professional fashion analysis for a clothing retailer. "
                f"SIMILAR ITEMS (always include this section in your response):\n{items_description}\n\n"
                "Please:\n"
                "1. Note these are similar but not exact items\n"
                "2. Identify clothing elements objectively (colors, patterns, materials)\n" 
                "3. Include the SIMILAR ITEMS section at the end\n\n"
                "This is for a professional retail catalog. Use formal, clinical language."
            )
        
        # Send the request to the model with the generated prompt
        response = self.generate_response(user_image_base64, assistant_prompt)
        
        # Check if the response is incomplete and create a basic response if necessary
        if len(response) < 100:
            logger.info("Response appears incomplete, creating basic response")
            # Create a basic response with the item details
            section_header = "ITEM DETAILS:" if similarity_score >= threshold else "SIMILAR ITEMS:"
            response = f"# Fashion Analysis\n\nThis outfit features a collection of carefully coordinated pieces.\n\n{section_header}\n{items_description}"
        
        # Ensure the response includes the item details section
        elif "ITEM DETAILS:" not in response and "SIMILAR ITEMS:" not in response:
            logger.info("Item details section missing from response")
            # Append to existing response
            section_header = "ITEM DETAILS:" if similarity_score >= threshold else "SIMILAR ITEMS:"
            response += f"\n\n{section_header}\n{items_description}"
        
        return response
