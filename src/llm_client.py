# llm_client.py

import os
from dotenv import load_dotenv
from groq import Groq
import requests

load_dotenv()

class LLMClient:
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or "llama-3.3-70b-versatile"
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        try:
            self.client = Groq(api_key=self.api_key)
            self.use_sdk = True
        except:
            self.use_sdk = False
    
    def chat(self, prompt, system_prompt=None, context=None):
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful logistics AI assistant."}
        ]
        
        if context:
            # Add retrieved context as system message
            messages.append({
                "role": "system",
                "content": "Here is relevant information from the freight database that might help answer the query:\n\n" + context
            })
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            if self.use_sdk:
                # Use the Groq SDK
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=0.7,
                    max_tokens=1000,
                )
                return chat_completion.choices[0].message.content
            else:
                # Use REST API directly
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM Error] {str(e)}"
