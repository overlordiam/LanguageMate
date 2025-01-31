import asyncio

from llm_inference_engine import LLMInferenceEngine


async def main():
    # Initialize the engine
    engine = LLMInferenceEngine()
    
    # Single inference
    response = await engine.generate_response("Can you explain me how ollama works?")
    print(response)
    
    # Batch inference
    # inputs = [
    #     "Hello, how are you?"
    #     "Bonjour, comment allez-vous?",
    #     "¿Hola, cómo estás?",
    #     "こんにちは",
    #     "안녕하세요"
    # ]
    # responses = await engine.batch_generate(inputs)
    # for resp in responses:
    #     print(resp)

if __name__ == "__main__":
    asyncio.run(main()) 