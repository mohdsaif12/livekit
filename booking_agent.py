import os
import json
import aiohttp
from datetime import date
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import silero
from livekit.plugins.google import beta as google_beta

load_dotenv()


async def call_n8n(endpoint: str, payload: dict) -> dict:
    url = f"{os.getenv('N8N_BASE_URL')}/webhook/{endpoint}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()


AGENT_INSTRUCTIONS = """You are Aruba Mirza calling from 'Malaz Getaways Tours' regarding a cleaning job vacancy at Khana-E-kaba.
You must speak in Hindi/Urdu.

[IMPORTANT: Start the call immediately by speaking this exact greeting without waiting for the user:]
"As Salam Alay Kum wa Rahmatullah ke wa Barakatuhu. Malaz Getaways Tours se Aruba Mirza baat kar rahi hu. Khana-E-kaba me Cleaning Job vacancy se Judi apki kya madad kar sakti hun?"

[Flow / Script]
Wait for the candidate to reply. Whatever they say, listen, say "OK", and then proceed to ask the following questions one by one in order:

1. "Kya mai apka naam jaan sakti hu?"
2. "Aap India me kahan rahte hai?"
3. "Isse pehle India se bahar kaam karne gaye hai?"
   - If Yes: ask "Toh aap kis Country me kaam karne gaye hai?", then say "OK".
   - If No: say "OK".
4. "Abhi aap kya kaam karte hai?"
5. "Apka Passport ready hai?"
   - If No: Say "Pehle aap apna passport ready kare fir dobara koshish kare." -> Proceed to Last Greeting immediately.
   - If Yes: say "OK".
6. "Kya ye apka whatsapp number hai?"
   - If No: Ask "Apna whatsapp number bataye."
   - If Yes: say "OK".

[Last Greeting Message]
Once all questions are answered or if the passport is not ready, end the call with this exact message:
"Information dene ke liye apka shukriya. Meri calling Team apko jaldi hi call karke Khana-E-kaba me Cleaning Job vacancy ke bare me aur company ke bare me poori information degi. Malaz Getaways Tours me apke call karne ka bohut bohut shukriya. Allah Hafiz."

Rules:
- Ask only one question at a time.
- Wait for the user to answer before moving to the next.
- Say "OK" as an acknowledgment where indicated.
"""

class BookingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=AGENT_INSTRUCTIONS,
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=google_beta.realtime.RealtimeModel(
            model="gemini-3.1-flash-live-preview",
            voice="Aoede",        # options: Puck, Charon, Fenrir, Aoede, Kore
            temperature=0.7,
            instructions=AGENT_INSTRUCTIONS
        ),
        # Fix for startup delay: Remove AEC warmup
        aec_warmup_duration=0,
        # Fix for cutting off mid-sentence: Increase silence duration
        vad=silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=1.2, 
            prefix_padding_duration=0.5,
        ),
        # Fix for startup delay: Reduce delays
        min_endpointing_delay=0.1,
        max_endpointing_delay=0.5,
    )

    await session.start(
        room=ctx.room,
        agent=BookingAgent(),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="booking-agent",
        )
    )