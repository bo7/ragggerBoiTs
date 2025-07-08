#!/usr/bin/env python3
"""
Test the unified LLM client with OpenRouter backend
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_unified_llm_client():
    """Test the unified LLM client"""
    
    # Load environment
    load_dotenv()
    
    # Check if API key is set
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ Please set your OpenRouter API key in the .env file")
        return False
    
    print("🚀 Testing Unified LLM Client with OpenRouter...")
    
    try:
        # Import the unified client
        from rag_system.llm_serving import LLMClient, LLMBackend
        
        # Initialize client with OpenRouter backend
        client = LLMClient(preferred_backend=LLMBackend.OPENROUTER)
        
        # Initialize with Mistral 7B free model
        print("📡 Initializing with Mistral 7B (free)...")
        success = await client.initialize(model_name="mistral-7b")
        
        if not success:
            print("❌ Failed to initialize unified LLM client")
            return False
        
        print("✅ Unified LLM client initialized successfully!")
        
        # Test health check
        print("\n🔍 Checking health status...")
        health = await client.health_check()
        print(f"Health status: {health}")
        
        # Test simple generation
        print("\n💬 Testing text generation...")
        response = await client.generate(
            "What is the capital of France? Answer in one word.",
            max_tokens=10,
            temperature=0.1
        )
        print(f"Response: {response}")
        
        # Test chat completion
        print("\n🗣️ Testing chat completion...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Keep responses concise."},
            {"role": "user", "content": "What is machine learning? Answer in one sentence."}
        ]
        
        chat_response = await client.chat(
            messages,
            max_tokens=50,
            temperature=0.7
        )
        print(f"Chat response: {chat_response}")
        
        # Test streaming
        print("\n🌊 Testing streaming generation...")
        print("Streaming response: ", end="", flush=True)
        
        stream_response = await client.generate(
            "Count from 1 to 3:",
            max_tokens=20,
            temperature=0.3,
            stream=True
        )
        
        async for chunk in stream_response:
            print(chunk, end="", flush=True)
        
        print("\n")
        
        # Test switching models
        print("\n🔄 Testing model switching...")
        if await client.switch_backend(LLMBackend.OPENROUTER, "deepseek-r1"):
            print("✅ Successfully switched to DeepSeek R1")
            
            deepseek_response = await client.generate(
                "What is 2+2?",
                max_tokens=10,
                temperature=0.1
            )
            print(f"DeepSeek response: {deepseek_response}")
        else:
            print("❌ Failed to switch to DeepSeek R1")
        
        # Test with newer models
        print("\n🆕 Testing with Llama 4 Maverick...")
        if await client.switch_backend(LLMBackend.OPENROUTER, "llama-4-maverick"):
            print("✅ Successfully switched to Llama 4 Maverick")
            
            llama4_response = await client.generate(
                "Explain quantum computing in one sentence.",
                max_tokens=30,
                temperature=0.5
            )
            print(f"Llama 4 response: {llama4_response}")
        else:
            print("❌ Failed to switch to Llama 4 Maverick")
        
        # Test available models
        print("\n📚 Available models:")
        models = client.get_available_models()
        openrouter_models = models.get("openrouter", {})
        for alias, model_id in list(openrouter_models.items())[:5]:  # Show first 5
            print(f"  {alias}: {model_id}")
        
        # Clean up
        await client.shutdown()
        print("\n✅ All unified LLM client tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 RAG System - Unified LLM Client Test")
    print("=" * 45)
    
    # Run the test
    success = asyncio.run(test_unified_llm_client())
    
    if success:
        print("\n🎉 Unified LLM client is working perfectly!")
        print("🚀 Ready to build the rest of the RAG system!")
    else:
        print("\n❌ Unified LLM client test failed.")


if __name__ == "__main__":
    main()