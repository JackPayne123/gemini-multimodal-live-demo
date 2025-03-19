# ... existing imports
from custom_gemini_service import FixedGeminiMultimodalLiveLLMService  # Add this import

# ... existing code

# Replace the original service with your custom one
# Instead of:
# llm_rt = GeminiMultimodalLiveLLMService(...)

# Use:
llm_rt = FixedGeminiMultimodalLiveLLMService(
    api_key=api_key,
    model=model_name,
    max_tokens=4096,
    temperature=0.2,
    # ... rest of your original parameters
)

# ... rest of your existing code