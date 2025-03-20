#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from asyncio import events
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
    GeminiMultimodalModalities,
    InputParams,
)
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

SYSTEM_INSTRUCTION = f"""
"You are Gemini Chatbot, a friendly, helpful robot.

Your goal is to demonstrate your capabilities in a succinct way.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""


async def main():
    try:
        # Capture all stdout
        import sys
        original_stdout = sys.stdout
        log_file = open("gemini_output.log", "w", encoding="utf-8")
        sys.stdout = log_file
        
        async with aiohttp.ClientSession() as session:
            (room_url, token) = await configure(session)

            transport = DailyTransport(
                room_url,
                token,
                "Respond bot",
                DailyParams(
                    audio_out_enabled=True,
                    vad_enabled=True,
                    vad_audio_passthrough=True,
                    transcription_enabled=True,
                    # set stop_secs to something roughly similar to the internal setting
                    # of the Multimodal Live api, just to align events. This doesn't really
                    # matter because we can only use the Multimodal Live API's phrase
                    # endpointing, for now.
                    vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
                ),
            )

            # Create a simpler frame processor for printing LLM responses
            class PrintLLMResponse(FrameProcessor):
                async def process_frames(self, frames):
                    try:
                        # Just print each frame for debugging
                        print(f"\nReceived {len(frames)} frames from LLM:")
                        for i, frame in enumerate(frames):
                            print(f"Frame {i} type: {type(frame).__name__}")
                            
                            # Try different ways to access the text content
                            # Method 1: Direct attribute access
                            if hasattr(frame, "text"):
                                print(f"  Text (attribute): {frame.text}")
                            
                            # Method 2: Dictionary-style access
                            try:
                                if isinstance(frame, dict) and "text" in frame:
                                    print(f"  Text (dict): {frame['text']}")
                            except (TypeError, KeyError):
                                pass
                            
                            # Method 3: __dict__ access
                            try:
                                if hasattr(frame, "__dict__") and "text" in frame.__dict__:
                                    print(f"  Text (__dict__): {frame.__dict__['text']}")
                            except (AttributeError, KeyError):
                                pass
                            
                            # Print all string attributes for inspection
                            print("  All string attributes:")
                            for attr in dir(frame):
                                if not attr.startswith("_") and not callable(getattr(frame, attr)):
                                    try:
                                        value = getattr(frame, attr)
                                        if isinstance(value, str):
                                            print(f"    {attr}: {value}")
                                    except Exception:
                                        pass
                        
                    except Exception as e:
                        print(f"ERROR in PrintLLMResponse: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    
                    return frames

            # Modify the Gemini configuration to ensure API key is valid and handle errors
            try:
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable is not set")
                
                # Add a timeout to prevent indefinite waiting
                llm = GeminiMultimodalLiveLLMService(
                    api_key=api_key,
                    transcribe_user_audio=True,
                    transcribe_model_audio=True,
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[{"google_search": {}}, {"code_execution": {}}],
                    params=InputParams(modalities=GeminiMultimodalModalities.TEXT),
                )
                
                print(f"Initialized Gemini service with API key: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else ''}")
            except Exception as e:
                print(f"ERROR initializing Gemini service: {str(e)}")
                raise

            #   Optionally, you can set the response modalities via a function
            #llm.set_model_modalities(
            #    GeminiMultimodalModalities.AUDIO
            #)

            tts = CartesiaTTSService(
                api_key=os.getenv("CARTESIA_API_KEY"), voice_id="71a7ad14-091c-4e8e-a314-022ece01c121"
            )

            messages = [
                {
                    "role": "user",
                    "content": 'Start by saying "Hello, I\'m Gemini".',
                },
            ]

            # Set up conversation context and management
            # The context_aggregator will automatically collect conversation context
            context = OpenAILLMContext(messages)
            context_aggregator = llm.create_context_aggregator(context)

            # Create a logging processor for the pipeline
            class GeminiResponseLogger(FrameProcessor):
                async def process_frames(self, frames):
                    try:
                        print("\n===== PROCESSING FRAMES FROM GEMINI =====")
                        print(f"Received {len(frames)} frames")
                        
                        for i, frame in enumerate(frames):
                            print(f"Frame {i} type: {type(frame).__name__}")
                            
                            if hasattr(frame, "text") and frame.text:
                                print(f"Text content: {frame.text}")
                            
                            if hasattr(frame, "search_result") and frame.search_result:
                                print(f"Search result: {frame.search_result}")
                                
                            # Print all frame attributes for debugging
                            for attr_name in dir(frame):
                                if not attr_name.startswith('_') and not callable(getattr(frame, attr_name)):
                                    attr_value = getattr(frame, attr_name)
                                    print(f"  {attr_name}: {attr_value}")
                        
                        print("========================================\n")
                    except Exception as e:
                        print(f"ERROR in GeminiResponseLogger: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    
                    # Pass frames through unchanged
                    return frames

            pipeline = Pipeline(
                [
                    transport.input(),
                    context_aggregator.user(),
                    llm,
                    #GeminiResponseLogger(),  # Add our new logger
                    tts,
                    transport.output(),
                    context_aggregator.assistant(),
                ]
            )

            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    allow_interruptions=True,
                    enable_metrics=True,
                    enable_usage_metrics=True,

                ),
            )

            @transport.event_handler("on_first_participant_joined")
            async def on_first_participant_joined(transport, participant):
                await transport.capture_participant_transcription(participant["id"])
                await task.queue_frames([context_aggregator.user().get_context_frame()])

            @transport.event_handler("on_participant_left")
            async def on_participant_left(transport, participant, reason):
                print(f"Participant left: {participant}")
                await task.cancel()

            runner = PipelineRunner()

            # Update the _receive_task_handler method to include more detailed logging
            async def _receive_task_handler_with_logging(self):
                print("Starting _receive_task_handler")
                try:
                    async for message in self._websocket:
                        print(f"Received raw message from websocket: {message[:200]}...")
                        
                        # Use the specific events module from the Gemini package
                        from pipecat.services.gemini_multimodal_live import events as gemini_events
                        evt = gemini_events.parse_server_event(message)
                        
                        print(f"Parsed event type: {type(evt).__name__}")
                        
                        if evt.setupComplete:
                            print("Handling setup complete event")
                            await self._handle_evt_setup_complete(evt)
                        elif evt.serverContent and evt.serverContent.modelTurn:
                            print("Handling model turn event")
                            await self._handle_evt_model_turn(evt)
                        elif evt.serverContent and evt.serverContent.turnComplete:
                            print("Handling turn complete event")
                            await self._handle_evt_turn_complete(evt)
                        elif evt.toolCall:
                            print("Handling tool call event")
                            await self._handle_evt_tool_call(evt)
                        else:
                            print(f"Unhandled event type: {evt}")
                except Exception as e:
                    print(f"ERROR in _receive_task_handler: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # Replace the original method with our logging version
            GeminiMultimodalLiveLLMService._receive_task_handler = _receive_task_handler_with_logging

            # Add a timeout to the main pipeline run
            async def run_with_timeout(task, timeout=60):
                print(f"Starting pipeline runner with {timeout} second timeout...")
                try:
                    await asyncio.wait_for(runner.run(task), timeout=timeout)
                    print("Pipeline run completed successfully")
                except asyncio.TimeoutError:
                    print(f"Pipeline execution timed out after {timeout} seconds")
                except Exception as e:
                    print(f"ERROR running pipeline: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # Use the timeout version
            # Modify the end of your main function to use this
            await run_with_timeout(task, timeout=120)  # 2 minute timeout

    except Exception as e:
        print(f"ERROR in main function: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        if 'log_file' in locals():
            log_file.close()


if __name__ == "__main__":
    asyncio.run(main())