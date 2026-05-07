
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
from livekit.plugins import openai, deepgram, silero
from livekit.plugins.sarvam import TTS

load_dotenv()

async def call_n8n(endpoint: str, payload: dict) -> dict:
    url = f"{os.getenv('N8N_BASE_URL')}/webhook/{endpoint}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()


class BookingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
[Identity]  
You are Neha, a warm and friendly receptionist at Time Coaching Center, helping clients with call bookings and their inquiries.

Default Language: Hindi (Devanagari).

If the caller starts speaking English, switch to English for the entire conversation.

Use casual vocabulary such as class, registration, courses, tutoring, services, and timing.

Working Hours: 9 AM – 7 PM (Monday to Saturday).

[Style]  
- Keep replies brief, between 8 to 12 words.  
- Maintain a natural, steady, and polite tone—avoid sounding robotic.  
- Break lengthy information into 2-3 concise sentences.  
- Match the caller’s mood (casual, busy, serious) appropriately.

[Response Guidelines]  
- Begin with a welcoming tone.
- Clarify the reason for the call promptly.
- Use simple and clear language in both Hindi and English as needed.
- Confirm twice for clarity on details.
- Use positive language, especially when handling booking confirmations or availability checks.

[Task & Goals]  
1. Greet the caller:

   - Hindi: “Time Coaching Center में आपका स्वागत है। मैं नेहा बोल रही हूँ. मैं आपकी किस प्रकार से सहायता कर सकती हूँ?”  
   - English: “Welcome to Bright Minds Coaching Center. This is Neha. How may I assist you?”

2. Identify the reason for the call:

   - Registration: “ठीक है, आप कौनसा course register करना चाहेंगे?”  
   - Inquiry: “जी, कौनसी जानकारी चाहिए आपको?”

3. Ask about their preferred timing:

   - “कब और कौनसा time आपके लिए सुविधाजनक रहेगा?”

4. Check availability:

   - If slots are available: “उस time पर class available है।”  
   - If not available: “उस time पर class full है। [Alternate time] available है। क्या ये सही रहेगा?”

5. Confirm details with the caller:

   - “तो confirm कर लूँ — आपकी class [course], [day], [time] पर है। सही है न?”

6. Proceed based on confirmation:

   - If yes: “Perfect, आपकी registration हो गई है। आपको SMS confirmation भी मिलेगा।”  
   - If no: “ठीक है, कोई बात नहीं। जब भी निर्णय लें, call कर लीजिए।”

7. Close the call appropriately:

   - If registered: “Thank you, [day/time] पर मिलते हैं Bright Minds Coaching Center में।”  
   - If not registered: “Thank you for calling Bright Minds Coaching Center. Have a nice day.”

[Error Handling / Fallback]  
- If unclear input, ask a clarifying question.
- Politely apologize if any system issue occurs and provide alternative solutions.

[Call Closing]  
- Always end with a clear goodbye.  
- Avoid lingering or asking, "are you still there?".
""",
        )

    @function_tool
    async def check_availability(self, context: RunContext, date_str: str, time_str: str) -> str:
        """Check if an appointment slot is available.
        Args:
            date_str: Date in YYYY-MM-DD format
            time_str: Time in HH:MM 24hr format
        """
        result = await call_n8n("check-availability", {"date": date_str, "time": time_str})
        return json.dumps(result)

    @function_tool
    async def book_appointment(self, context: RunContext, name: str, phone: str, appointment_type: str, date_str: str, time_str: str, notes: str = "") -> str:
        """Book a confirmed appointment.
        Args:
            name: Customer full name
            phone: Customer phone number
            date_str: Date in YYYY-MM-DD format
            time_str: Time in HH:MM 24hr format
            notes: Any special notes (optional)
        """
        result = await call_n8n("book-appointment", {"name": name, "phone": phone, "type": appointment_type, "date": date_str, "time": time_str, "notes": notes})
        return json.dumps(result)

    @function_tool
    async def reschedule_appointment(self, context: RunContext, booking_id: str, new_date: str, new_time: str) -> str:
        """Reschedule an existing appointment.
        Args:
            booking_id: Existing booking ID
            new_date: New date in YYYY-MM-DD format
            new_time: New time in HH:MM 24hr format
        """
        result = await call_n8n("reschedule", {"booking_id": booking_id, "new_date": new_date, "new_time": new_time})
        return json.dumps(result)

    @function_tool
    async def cancel_appointment(self, context: RunContext, booking_id: str, reason: str = "") -> str:
        """Cancel an existing appointment.
        Args:
            booking_id: Booking ID to cancel
            reason: Reason for cancellation (optional)
        """
        result = await call_n8n("cancel", {"booking_id": booking_id, "reason": reason})
        return json.dumps(result)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(
           model="nova-2-general",
           language="multi",
           smart_format=True,
           punctuate=True,
           interim_results=True,
           filler_words=False,
           endpointing_ms=50,
        ),
        llm=openai.LLM(
            model="gpt-4o-mini",
        ),
        tts=TTS(
            model="bulbul:v2",
            target_language_code="hi-IN",
            speaker="anushka",
            speech_sample_rate=22050,
            pace=1.0,
        ),
        vad=silero.VAD.load(
            min_speech_duration=0.03,
        min_silence_duration=0.08,
        prefix_padding_duration=0.05,
        ),
        min_endpointing_delay=0.0,
        max_endpointing_delay=0.2,
    )

    await session.start(
        room=ctx.room,
        agent=BookingAgent(),
    )

    await session.generate_reply(
        instructions="Introduce yourself as Priya in Hindi and ask how you can help with booking."
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="booking-agent",
        )
    )
