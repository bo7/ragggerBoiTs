#!/usr/bin/env python3
"""
Test script for Jina v4 embeddings API
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from rag_system.embedding import JinaEmbeddingClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_jina_embeddings():
    """Test the Jina embeddings client"""
    
    # Load environment
    load_dotenv()
    
    # Check if API key is set
    api_key = os.getenv('JINA_API_KEY')
    if not api_key or api_key == 'your_jina_api_key_here':
        print("❌ Please set your Jina API key in the .env file")
        print("   Get your API key from: https://jina.ai/embeddings/")
        print("   New users get 10 million free tokens!")
        return False
    
    print("🚀 Testing Jina v4 Embeddings API...")
    print(f"Using API key: {api_key[:20]}...")
    
    try:
        # Initialize client with Jina v4
        client = JinaEmbeddingClient(
            api_key=api_key,
            model_name="jina-embeddings-v4"
        )
        
        # Initialize client
        print("\n📡 Initializing Jina embedding client...")
        success = await client.initialize()
        
        if not success:
            print("❌ Failed to initialize Jina embedding client")
            return False
        
        print("✅ Jina embedding client initialized successfully!")
        
        # Test model info
        print("\n📋 Getting model information...")
        model_info = await client.get_model_info()
        print(f"Model info: {model_info}")
        
        # Test simple text embedding
        print("\n📝 Testing text embedding...")
        test_texts = [
            "Hello, world!",
            "This is a test of the Jina embeddings API.",
            "Machine learning is transforming how we process information."
        ]
        
        text_embeddings = await client.embed_text(
            test_texts,
            task="text-matching"
        )
        
        print(f"✅ Text embeddings generated!")
        print(f"   - Number of embeddings: {len(text_embeddings)}")
        print(f"   - Embedding dimension: {len(text_embeddings[0]) if text_embeddings else 0}")
        print(f"   - First embedding (first 5 values): {text_embeddings[0][:5] if text_embeddings else 'None'}")
        
        # Test retrieval tasks
        print("\n🔍 Testing retrieval tasks...")
        
        # Query embedding
        query_embedding = await client.embed_text(
            "What is machine learning?",
            task="retrieval.query"
        )
        
        # Passage embedding
        passage_embedding = await client.embed_text(
            "Machine learning is a subset of artificial intelligence that uses algorithms to learn from data.",
            task="retrieval.passage"
        )
        
        # Compute similarity
        similarity = await client.compute_similarity(
            query_embedding[0],
            passage_embedding[0],
            metric="cosine"
        )
        
        print(f"✅ Retrieval task completed!")
        print(f"   - Query embedding dimension: {len(query_embedding[0])}")
        print(f"   - Passage embedding dimension: {len(passage_embedding[0])}")
        print(f"   - Cosine similarity: {similarity:.4f}")
        
        # Test code embeddings (if supported)
        print("\n💻 Testing code embeddings...")
        
        code_query = await client.embed_text(
            "How to sort a list in Python?",
            task="code.query"
        )
        
        code_passage = await client.embed_text(
            "def sort_list(lst): return sorted(lst)",
            task="code.passage"
        )
        
        code_similarity = await client.compute_similarity(
            code_query[0],
            code_passage[0],
            metric="cosine"
        )
        
        print(f"✅ Code embeddings generated!")
        print(f"   - Code query dimension: {len(code_query[0])}")
        print(f"   - Code passage dimension: {len(code_passage[0])}")
        print(f"   - Code similarity: {code_similarity:.4f}")
        
        # Test batch processing
        print("\n📦 Testing batch processing...")
        
        batch_texts = [
            "First document about artificial intelligence",
            "Second document about machine learning",
            "Third document about deep learning",
            "Fourth document about neural networks",
            "Fifth document about data science"
        ]
        
        batch_embeddings = await client.batch_embed(
            batch_texts,
            batch_size=3,
            task="retrieval.passage"
        )
        
        print(f"✅ Batch processing completed!")
        print(f"   - Batch size: {len(batch_embeddings)}")
        print(f"   - Embedding dimension: {len(batch_embeddings[0]) if batch_embeddings else 0}")
        
        # Test late chunking (if supported)
        print("\n🧩 Testing late chunking...")
        
        long_text = " ".join([
            "This is a very long document that will be processed with late chunking.",
            "Late chunking allows for better handling of long documents by processing them in chunks.",
            "This feature is particularly useful for documents that exceed the model's maximum sequence length.",
            "The Jina v4 model supports late chunking for improved performance on long texts."
        ] * 10)  # Make it longer
        
        chunked_embeddings = await client.embed_text(
            long_text,
            task="retrieval.passage",
            late_chunking=True
        )
        
        print(f"✅ Late chunking completed!")
        print(f"   - Text length: {len(long_text)} characters")
        print(f"   - Embedding dimension: {len(chunked_embeddings[0]) if chunked_embeddings else 0}")
        
        # Test supported models
        print("\n📚 Supported models:")
        supported_models = client.get_supported_models()
        for model_name, config in supported_models.items():
            print(f"   - {model_name}:")
            print(f"     • Dimension: {config['dimension']}")
            print(f"     • Max sequence length: {config['max_seq_length']}")
            print(f"     • Multimodal support: {config['supports_multimodal']}")
            print(f"     • Late chunking: {config['supports_late_chunking']}")
            print(f"     • Supported tasks: {config['supports_tasks']}")
            print()
        
        # Test with different model
        print("\n🔄 Testing with Jina v3 model...")
        if await client.switch_model("jina-embeddings-v3"):
            print("✅ Successfully switched to Jina v3")
            
            v3_embedding = await client.embed_text(
                "Testing with Jina v3 model",
                task="text-matching"
            )
            print(f"   - Jina v3 embedding dimension: {len(v3_embedding[0]) if v3_embedding else 0}")
        else:
            print("❌ Failed to switch to Jina v3")
        
        # Test pricing info
        print("\n💰 Pricing information:")
        pricing_info = client.get_pricing_info()
        for key, value in pricing_info.items():
            print(f"   - {key}: {value}")
        
        # Test embedding test
        print("\n🧪 Running embedding test...")
        test_result = await client.test_embedding(
            "This is a test of the Jina embedding service",
            task="text-matching"
        )
        
        if test_result['success']:
            print("✅ Embedding test passed!")
            print(f"   - Response time: {test_result['response_time']:.3f}s")
            print(f"   - Embedding dimension: {test_result['embedding_dimension']}")
        else:
            print(f"❌ Embedding test failed: {test_result['error']}")
        
        # Clean up
        await client.shutdown()
        print("\n✅ All Jina embedding tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 RAG System - Jina v4 Embeddings Test")
    print("=" * 40)
    
    # Run the test
    success = asyncio.run(test_jina_embeddings())
    
    if success:
        print("\n🎉 Jina v4 embeddings are working perfectly!")
        print("🚀 Ready to integrate with the RAG system!")
    else:
        print("\n❌ Jina embeddings test failed.")
        print("Please check your API key and try again.")


if __name__ == "__main__":
    main()