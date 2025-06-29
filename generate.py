import google.generativeai as genai
import asyncio
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_random
import json

from retrieve import DocumentRetriever

class Gemini:
    def __init__(self, model_name: str):
        self.model = model_name
        self.api_key = "api-key" # TODO: set the api key 
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model_name=self.model)
        
    async def query_async(self, prompt):
        response = await self.client.generate_content_async(prompt)
        return response.text
    
    def query(self, prompt):
        '''
        This function is a wrapper around the Gemini API.
        It takes a string prompt and returns a response from Gemini.
        '''
        response = self.client.generate_content(prompt)
        return response.text



class Generator:
    def __init__(self):
        self.model = Gemini("gemini-1.5-flash-latest")
        self.dr = DocumentRetriever()


    def process_query(self, query, documents):
        
        prompt = f"""
        You are a medical AI assistant. 
        Given the patient's medical question and the relevant medical documents,
        generate a well-structured, informative response.

        Ensure your answer is clear, concise, and directly addresses the patient's question, 
        citing relevant information from the provided documents.


        Patient's question: 
        "{query}"

        Relevant medical documents:
        {documents}

        Your response should be a json object in the following format:
        {{   
            "answer":<answer>
        }}
        
        It is imperative that you only return the json object in the following format or else you will be penalized.
        """

        
        # prompt = prompt.format(query = query, documents = documents) 
        prompt = self.optimize_query(prompt) 
        response = self.model.query(prompt) 
        response = response[response.find("{"):response.find("}")+1]
        response = json.loads(response)
        return response['answer']
    
    # def optimize_query(self, original_query):
    #     optimized_query = None 
    #     return original_query if optimized_query is None else optimized_query
    
