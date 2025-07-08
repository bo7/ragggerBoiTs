#!/usr/bin/env python3
"""
Test script for the LLM client with OpenRouter
"""

import asyncio
import logging
import os
from rag_system.llm_serving import LLMClient, LLMBackend

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_openrouter_client():
    """Test the OpenRouter LLM client"""
    
    # Check if API key is set
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ Please set your OpenRouter API key in the .env file")
        print("   Get your API key from: https://openrouter.ai/keys")
        return False
    
    print("🚀 Testing OpenRouter LLM Client...")
    
    try:
        # Initialize client with OpenRouter backend
        client = LLMClient(preferred_backend=LLMBackend.OPENROUTER)
        
        # Initialize with a free model
        print("📡 Initializing with Llama 3.2 7B (free model)...")
        success = await client.initialize(model_name="llama-3.2-7b")
        
        if not success:
            print("❌ Failed to initialize LLM client")
            return False
        
        print("✅ LLM client initialized successfully!")
        
        # Test health check
        print("\n🔍 Checking health status...")
        health = await client.health_check()
        print(f"Health status: {health}")
        
        # Test simple generation
        print("\n💬 Testing text generation...")
        response = await client.generate(
            "Hello! Please introduce yourself briefly.",
            max_tokens=100,
            temperature=0.7
        )
        print(f"Response: {response}")
        
        # Test chat completion
        print("\n🗣️ Testing chat completion...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the benefits of open-source AI models?"}
        ]
        
        chat_response = await client.chat(
            messages,
            max_tokens=150,
            temperature=0.7
        )
        print(f"Chat response: {chat_response}")
        
        # Test streaming
        print("\n🌊 Testing streaming generation...")
        print("Streaming response: ", end="", flush=True)
        
        async for chunk in await client.generate(
            "Count from 1 to 5, explaining each number:",
            max_tokens=100,
            temperature=0.3,
            stream=True
        ):
            print(chunk, end="", flush=True)
        
        print("\n")
        
        # Test model switching
        print("\n🔄 Testing model switching...")
        if await client.switch_backend(LLMBackend.OPENROUTER, "mistral-7b"):
            print("✅ Successfully switched to Mistral 7B")
            
            mistral_response = await client.generate(
                "What is machine learning in one sentence?",
                max_tokens=50,
                temperature=0.5
            )
            print(f"Mistral response: {mistral_response}")
        else:
            print("❌ Failed to switch to Mistral 7B")
        
        # Clean up
        await client.shutdown()
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False


async def test_model_availability():
    """Test available models"""
    print("\n📋 Available OpenRouter models:")
    
    from rag_system.llm_serving.openrouter_client import OpenRouterClient
    
    models = OpenRouterClient.get_available_models()
    for alias, model_id in models.items():
        print(f"  {alias}: {model_id}")
    
    print("\n🆓 Free models:")
    free_models = OpenRouterClient.get_free_models()
    for alias, model_id in free_models.items():
        print(f"  {alias}: {model_id}")


def main():
    """Main test function"""
    print("🧪 RAG System - LLM Client Test")
    print("=" * 40)
    
    # Test model availability first
    asyncio.run(test_model_availability())
    
    # Test the actual client
    success = asyncio.run(test_openrouter_client())
    
    if success:
        print("\n🎉 All tests passed! The LLM client is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the configuration.")


if __name__ == "__main__":
    main()