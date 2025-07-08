#!/usr/bin/env python3
"""
List available OpenRouter models
"""

import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

async def list_models():
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            
            print(f"Found {len(models)} models")
            
            # Look for free models
            free_models = []
            llama_models = []
            mistral_models = []
            
            for model in models:
                model_id = model.get("id", "")
                pricing = model.get("pricing", {})
                
                # Check if it's free (price is 0)
                prompt_price = float(pricing.get("prompt", "1"))
                completion_price = float(pricing.get("completion", "1"))
                
                if prompt_price == 0 and completion_price == 0:
                    free_models.append(model_id)
                
                if "llama" in model_id.lower():
                    llama_models.append(model_id)
                
                if "mistral" in model_id.lower():
                    mistral_models.append(model_id)
            
            print("\n🆓 FREE MODELS:")
            for model in free_models:
                print(f"  {model}")
            
            print(f"\n🦙 LLAMA MODELS (first 10):")
            for model in llama_models[:10]:
                print(f"  {model}")
            
            print(f"\n🌟 MISTRAL MODELS (first 10):")
            for model in mistral_models[:10]:
                print(f"  {model}")
                
        else:
            print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    asyncio.run(list_models())