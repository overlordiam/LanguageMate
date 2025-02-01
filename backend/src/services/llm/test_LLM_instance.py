import asyncio

from backend.src.services.llm.llm_inference_engine import LLMInferenceEngine


async def main():
    # Initialize the engine
    engine = LLMInferenceEngine()
    
    # Single inference
    response = engine.generate_response("Bonjour, comment allez-vous?")
    print(response)
    
    # # Batch inference
    # inputs = [
    #     # "Hello, how are you?"
    #     # "Bonjour, comment allez-vous?",
    #     "¿Hola, cómo estás?"
    # ]
    # responses = await engine.batch_generate(inputs)
    # for resp in responses:
    #     print(resp)

if __name__ == "__main__":
    asyncio.run(main()) 