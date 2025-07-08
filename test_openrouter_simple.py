#!/usr/bin/env python3
"""
Simple test script for OpenRouter client only
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import just what we need for OpenRouter
from rag_system.llm_serving.openrouter_client import OpenRouterClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_openrouter_simple():
    """Simple test of OpenRouter client"""
    
    # Check if API key is set
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ Please set your OpenRouter API key in the .env file")
        print("   Get your API key from: https://openrouter.ai/keys")
        return False
    
    print("🚀 Testing OpenRouter Client...")
    print(f"Using API key: {api_key[:20]}...")
    
    try:
        # Initialize client with free model
        client = OpenRouterClient(
            api_key=api_key,
            model_name="mistralai/mistral-7b-instruct:free"
        )
        
        # Test health check
        print("\n🔍 Testing connection...")
        if await client.health_check():
            print("✅ Connection successful!")
        else:
            print("❌ Connection failed!")
            return False
        
        # Test simple generation
        print("\n💬 Testing text generation...")
        response = await client.generate(
            "Hello! Please respond with exactly 'Hello, World!' and nothing else.",
            max_tokens=10,
            temperature=0.1
        )
        print(f"Response: {response}")
        
        # Test chat completion
        print("\n🗣️ Testing chat completion...")
        messages = [
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ]
        
        chat_response = await client.chat(
            messages,
            max_tokens=5,
            temperature=0.1
        )
        print(f"Chat response: {chat_response}")
        
        # Test model info
        print("\n📋 Getting model info...")
        model_info = await client.get_model_info()
        print(f"Model info: {model_info}")
        
        # Test available models
        print("\n📖 Available models:")
        models = client.get_available_models()
        for alias, model_id in list(models.items())[:5]:  # Show first 5
            print(f"  {alias}: {model_id}")
        
        # Test free models
        print("\n🆓 Free models:")
        free_models = client.get_free_models()
        for alias, model_id in free_models.items():
            print(f"  {alias}: {model_id}")
        
        print("\n✅ All tests passed! OpenRouter is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 OpenRouter Client Test")
    print("=" * 30)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the test
    success = asyncio.run(test_openrouter_simple())
    
    if success:
        print("\n🎉 OpenRouter client is working perfectly!")
    else:
        print("\n❌ OpenRouter test failed. Please check your API key and connection.")


if __name__ == "__main__":
    main()