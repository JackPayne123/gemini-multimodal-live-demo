from typing import Any, Dict, Optional
from datetime import datetime

from bots.persistent_context import PersistentContext
from bots.rtvi import create_rtvi_processor
from bots.types import BotCallbacks, BotConfig, BotParams
from common.config import SERVICE_API_KEYS
from common.models import Conversation, Message
from loguru import logger
from openai._types import NOT_GIVEN
from sqlalchemy.ext.asyncio import AsyncSession
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from bots.webrtc.custom_gemini_service import FixedGeminiMultimodalLiveLLMService  # Add this import
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import (
    RTVIBotLLMProcessor,
    RTVIBotTranscriptionProcessor,
    RTVIBotTTSProcessor,
    RTVISpeakingProcessor,
    RTVIUserTranscriptionProcessor,
)
from pipecat.services.google import GoogleLLMService, LLMSearchResponseFrame

from pipecat.services.ai_services import OpenAILLMContext
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
    InputParams,
    GeminiMultimodalModalities
)
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.adapters.schemas.function_schema import FunctionSchema
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.transports.services.daily import DailyParams, DailyTransport
from vertexai import rag
from pipecat.frames.frames import TranscriptionFrame, TextFrame, Frame
from pipecat.transports.services.daily import TransportMessageUrgentFrame

async def retrieve_information(function_name, tool_call_id, args, llm, context, result_callback):
    """Retrieve information from Vertex RAG corpus based on a query."""
    try:
        query = args.get("query", "")
        
        # Optional start notification
        #await llm.push_frame(TTSSpeakFrame("Let me look that up for you."))
        
        # Set up RAG resource
        rag_resource = rag.RagResource(
            rag_corpus="projects/sandbox-ampelic/locations/us-central1/ragCorpora/6838716034162098176",
        )
        print("running retrieval query")
        
        # Perform retrieval query
        response = rag.retrieval_query(
            rag_resources=[rag_resource],
            text=query,
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=5,
                filter=rag.Filter(
                    vector_distance_threshold=0.5,
                ),
            ),
        )
        print(f"Retrieved context: {response}")
        # Process the retrieved context
        retrieved_context = " ".join(
            [context.text for context in response.contexts.contexts]
        ).replace("\n", " ")

        print(f"Retrieved context: {retrieved_context}")
        
        await result_callback(
            {
                "query": query,
                "information": retrieved_context,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
        )
    except Exception as e:
        print(f"Error in retrieve_information: {str(e)}")
        await result_callback(
            {
                "query": args.get("query", ""),
                "information": "Sorry, I couldn't retrieve the information due to an error.",
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
        )


# Proper async start callback for weather function
async def start_fetch_weather(function_name, llm, context):
    """Push a frame to the LLM to acknowledge the weather request."""
    await llm.push_frame(TTSSpeakFrame("Let me check on that weather for you."))
    logger.debug(f"Starting weather fetch: {function_name}")

# Proper async start callback for database query function
async def start_database_query(function_name, llm, context):
    """Push a frame to the LLM to acknowledge the database query."""
    await llm.push_frame(TTSSpeakFrame("Looking that up in our database."))
    logger.debug(f"Starting database query: {function_name}")

# Main function handler for weather with proper return format
async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    try:
        location = args.get("location", "unknown location")
        temperature_format = args.get("format", "celsius")
        temperature = 100 if temperature_format == "fahrenheit" else 37
        
        logger.debug(f"Fetching weather for {location} in {temperature_format}")
        
        # Return a properly structured object, not a string
        weather_data = {
            "conditions": "nice",
            "temperature": temperature,
            "format": temperature_format,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        
        await result_callback(weather_data)
    except Exception as e:
        error_message = f"Error fetching weather: {str(e)}"
        logger.error(error_message)
        await result_callback({"error": error_message})

# Define the function for database querying with proper error handling
async def query_database(function_name, tool_call_id, args, llm, context, result_callback):
    """Handle database query requests from the LLM"""
    try:
        # Parse the query from args
        query = args.get("query")
        if not query:
            await result_callback({"error": "No query provided"})
            return
            
        logger.debug(f"Executing database query: {query}")
        
        # IMPORTANT: Return a properly structured object, not a string
        results = {
            "results": "the best wine is y series red cabernet",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "query": query
        }
        
        # Return results as a structured object
        await result_callback(results)
        
    except Exception as e:
        error_message = f"Error executing database query: {str(e)}"
        logger.error(error_message)
        await result_callback({"error": error_message})


async def query_database_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    # Query database
    results = "the best wine is y series red cabernet"

    #async def on_update():
     #   await notify_system("Database query complete")

    # Run LLM after function call and notify when context is updated
    await result_callback(
        {
            "INFORMATION": results,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
        #properties=properties
    )


class SendMessageFrame(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._name = "message"
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)
            
            if isinstance(frame, TranscriptionFrame):
                # User speech transcription
                data = {"text": frame.text, "sender": "user"}
                message = TransportMessageUrgentFrame(message={"type": "message", "data": data})
                await self.push_frame(message)
            elif isinstance(frame, TextFrame):
                # LLM response text
                data = {"text": frame.text, "sender": "robot"}
                message = TransportMessageUrgentFrame(message={"type": "message", "data": data})
                await self.push_frame(message)
                
            await self.push_frame(frame, direction)
        except Exception as e:
            logger.debug(f"Exception in SendMessageFrame: {e}")


async def bot_pipeline(
    params: BotParams,
    config: BotConfig,
    callbacks: BotCallbacks,
    room_url: str,
    room_token: str,
    db: AsyncSession,
    text_only_mode: bool = False,  # Add text_only_mode parameter
) -> Pipeline:
    transport = DailyTransport(
        room_url,
        room_token,
        "Gemini Bot",
        DailyParams(
            audio_in_sample_rate=16000,
            audio_out_enabled=not text_only_mode,  # Disable audio output in text-only mode
            audio_out_sample_rate=24000,
            #transcription_enabled=True,  # Explicitly enable transcription
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            vad_audio_passthrough=True,
        ),
    )

    conversation = await Conversation.get_conversation_by_id(params.conversation_id, db)
    if not conversation:
        raise Exception(f"Conversation {params.conversation_id} not found")
    messages = [getattr(msg, "content") for msg in conversation.messages]

    # Define function schemas
    weather_function = FunctionSchema(
        name="get_current_weather",
        description="Get the current weather",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use.",
            },
        },
        required=["location", "format"],
    )
    
    db_query_function = FunctionSchema(
        name="query_database_from_api",
        description="Query the database for information about wines, food, and other products",
        properties={
            "query": {
                "type": "string", 
                "description": "The SQL query to execute or a natural language description of what data to retrieve",
            }
        },
        required=["query"],
    )
    
    retrieve_function = FunctionSchema(
            name="retrieve_information",
            description="Retrieve information from knowledge base",
            properties={
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve information about",
                },
            },
            required=["query"],
    )
    
    # Create tools schema with both functions
    tools_schema = ToolsSchema(
        standard_tools=[weather_function, 
                        #db_query_function,
                        retrieve_function
                        ]
    )



    # System instruction to guide the bot
    system_instruction = """
    You are a wine expert with vast knowledge of food and drink pairings and details about tasting notes and other interesting aspects of wine. 
    
    You can also retrieve information using the "retrieve_information" tool. If the user asks a question that requires
    looking up specific information, use this tool to find relevant data.

    If the user says something like "Can you tell me when X happened" use an appropriate query in the 
    and respond to the user with the new info you've retrieved. 

    Any info you retrieve from the tool, do NOT directly repeat it to the user. 
    Use that new info to inform your responses and recommendations about what the user is asking.


    """
    ##Before calling the tool, make sure to tell the user that you're checking.
    #You can use the "retrieve_information" tool to l

    ##You are a helpful assistant who can answer questions and use tools.

    # Configure Gemini service with appropriate modality
    if text_only_mode:
        llm_rt = FixedGeminiMultimodalLiveLLMService(
            api_key=str(SERVICE_API_KEYS["gemini"]),
            voice_id="Charon",
            transcribe_user_audio=True,
            transcribe_model_audio=True,
            tools=tools_schema,
            system_instruction=system_instruction,
            params=InputParams(modalities=GeminiMultimodalModalities.TEXT),  # Set text-only modality
        )
        # Print a note about text-only mode
        print("Running in text-only mode. Audio output disabled.")
    else:
        llm_rt = FixedGeminiMultimodalLiveLLMService(
            api_key=str(SERVICE_API_KEYS["gemini"]),
            voice_id="Charon",  # Puck, Charon, Kore, Fenrir, Aoede
            transcribe_user_audio=True,
            transcribe_model_audio=True,
            tools=tools_schema,
            system_instruction=system_instruction,
        )

    # Store db session on the LLM service for access from function handlers
    #llm_rt.db_session = db
    
    # Register function handlers with proper async start callbacks
    llm_rt.register_function(
        "get_current_weather",
        fetch_weather_from_api,
        start_callback=start_fetch_weather
    )
    
    llm_rt.register_function(
        "query_database_from_api",
        query_database_from_api,
        #start_callback=start_database_query
    )
    llm_rt.register_function("retrieve_information", retrieve_information)

    # Create context
    context_rt = OpenAILLMContext(messages)
    context_aggregator_rt = llm_rt.create_context_aggregator(context_rt)
    user_aggregator = context_aggregator_rt.user()
    assistant_aggregator = context_aggregator_rt.assistant()
    await llm_rt.set_context(context_rt)
    storage = PersistentContext(context=context_rt)

    rtvi = await create_rtvi_processor(config, user_aggregator)

    # This will send `user-*-speaking` and `bot-*-speaking` messages.
    rtvi_speaking = RTVISpeakingProcessor()

    # This will send `user-transcription` messages.
    rtvi_user_transcription = RTVIUserTranscriptionProcessor()

    # This will send `bot-transcription` messages.
    rtvi_bot_transcription = RTVIBotTranscriptionProcessor()

    # This will send `bot-llm-*` messages.
    rtvi_bot_llm = RTVIBotLLMProcessor()

    # This will send `bot-tts-*` messages.
    rtvi_bot_tts = RTVIBotTTSProcessor(direction=FrameDirection.UPSTREAM)

    # Create message senders for user and robot messages
    user_send = SendMessageFrame()
    robot_send = SendMessageFrame()
    
    # Build processors list with message senders
    processors = [
        transport.input(),
        rtvi,
        user_aggregator,
        #user_send,  # Add user message sender
        llm_rt,
        rtvi_speaking,
        rtvi_bot_llm,
        robot_send,  # Add robot message sender
        transport.output(),
        assistant_aggregator,
        storage.create_processor(exit_on_endframe=True),
    ]


    pipeline = Pipeline(processors)

    @storage.on_context_message
    async def on_context_message(messages: list[Any]):
        logger.debug(f"{len(messages)} message(s) received for storage")
        try:
            await Message.create_messages(
                db_session=db, conversation_id=params.conversation_id, messages=messages
            )
        except Exception as e:
            logger.error(f"Error storing messages: {e}")
            raise e

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        for message in params.actions:
            await rtvi.handle_message(message)

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        # Enable both camera and screenshare. From the client side
        # send just one.
        await transport.capture_participant_video(
            participant["id"], framerate=1, video_source="camera"
        )
        await transport.capture_participant_video(
            participant["id"], framerate=1, video_source="screenVideo"
        )
        await callbacks.on_first_participant_joined(participant)

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        await callbacks.on_participant_joined(participant)

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await callbacks.on_participant_left(participant, reason)

    @transport.event_handler("on_call_state_updated")
    async def on_call_state_updated(transport, state):
        await callbacks.on_call_state_updated(state)

    return pipeline
