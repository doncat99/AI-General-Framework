import asyncio

from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from google.adk.agents import LlmAgent as Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config import config


GoogleADKInstrumentor().instrument()

# Define tool function
def say_hello():
    return {"greeting": "Hello Langfuse 👋"}

# Configure agent
agent = Agent(
    name="hello_agent",
    model=LiteLlm(
        # Specify the OpenRouter model using 'openrouter/' prefix
        # model="openrouter/anthropic/claude-sonnet-4",
        model="openrouter/openai/gpt-4.1-mini",
        # Explicitly provide the API key from environment variables
        api_key=config.OPENROUTER_API_KEY,
        # Explicitly provide the OpenRouter API base URL
        api_base=config.OPENROUTER_BASE_URL,
    ),
    instruction="Always greet using the say_hello tool.",
    tools=[say_hello],
)

APP_NAME = "hello_app"
USER_ID = "demo-user"
SESSION_ID = "demo-session"

# Initialize session service
session_service = InMemorySessionService()

# Synchronous wrapper for session creation
def create_session_sync():
    try:
        asyncio.run(
            session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
            )
        )
    except Exception as e:
        print(f"Failed to create session: {e}")
        exit(1)

# Create session
create_session_sync()

# Initialize runner
runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

# Run agent
user_msg = types.Content(role="user", parts=[types.Part(text="hi")])
try:
    for event in runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg):
        if event.is_final_response():
            print(event.content.parts[0].text)
except Exception as e:
    print(f"Error running agent: {e}")