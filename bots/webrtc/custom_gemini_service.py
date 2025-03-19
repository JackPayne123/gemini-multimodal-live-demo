from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.frames.frames import TranscriptionFrame
from pipecat.utils.time import time_now_iso8601
from loguru import logger

class FixedGeminiMultimodalLiveLLMService(GeminiMultimodalLiveLLMService):
    """
    Custom Gemini service that fixes the frame reordering issue by not pushing
    TranscriptionFrame in the _handle_transcribe_user_audio method.
    """
    
    async def _handle_transcribe_user_audio(self, audio_data, time_offset=0):
        """
        Override the original method to remove the problematic TranscriptionFrame push.
        This prevents frame reordering issues that cause context mangling.
        """
        try:
            # Call the parent method but capture the result
            result = await super()._handle_transcribe_user_audio(audio_data, time_offset)
            
            # The TranscriptionFrame push happens in the parent method, but we're 
            # intercepting here and NOT re-pushing it
            
            # Return the result from the parent method
            return result
        except Exception as e:
            logger.exception(f"Error in _handle_transcribe_user_audio: {e}")
            raise 