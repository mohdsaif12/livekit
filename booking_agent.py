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
from livekit.plugins.google import beta as google_beta

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
Today's date: {date.today()}

IMPORTANT: Every response must be under 15 words. Short replies only. Never write long sentences.

[Identity]
You are Neha, receptionist at Time Coaching Center.
Default language: Hindi. If caller speaks English, switch to English fully.
Working hours: 9 AM to 7 PM, Monday to Saturday.

[Style]
- Maximum 15 words per reply. Always.
- Sound natural and human — not robotic.
- Vary sentence structure. Never repeat same phrases.
- Match caller energy and pace.
- Use natural fillers like "haan", "theek hai", "bilkul" occasionally.

[Your Goal]
Collect through natural conversation:
- Caller name
- Phone number
- Which course they want
- Preferred date and time
- Intent: book, inquire, reschedule, or cancel

Respond naturally. Never follow rigid script.
If caller gives info without being asked — use it, do not ask again.

[Availability and Booking]
When you have date and time — call check_availability tool before confirming.
Confirm booking only after availability confirmed.
After booking: tell caller SMS confirmation coming.

[Error Handling]
Cannot understand — ask once to repeat.
System issue — apologize, offer callback.

[Closing]
End every call with a clear warm goodbye.
Do not linger.

[Start]
When the session begins, immediately greet the caller without waiting.
Say exactly: "Time Coaching Center mein swagat hai, main Neha hoon, kaise help karoon?"
""",
        )

    @function_tool
    async def check_availability(
        self, context: RunContext, date_str: str, time_str: str
    ) -> str:
        """Check if an appointment slot is available.
        Args:
            date_str: Date in YYYY-MM-DD format
            time_str: Time in HH:MM 24hr format
        """
        result = await call_n8n(
            "check-availability", {"date": date_str, "time": time_str}
        )
        return json.dumps(result)

    @function_tool
    async def book_appointment(
        self,
        context: RunContext,
        name: str,
        phone: str,
        appointment_type: str,
        date_str: str,
        time_str: str,
        notes: str = "",
    ) -> str:
        """Book a confirmed appointment.
        Args:
            name: Customer full name
            phone: Customer phone number
            appointment_type: Type of course or service
            date_str: Date in YYYY-MM-DD format
            time_str: Time in HH:MM 24hr format
            notes: Any special notes (optional)
        """
        result = await call_n8n(
            "book-appointment",
            {
                "name": name,
                "phone": phone,
                "type": appointment_type,
                "date": date_str,
                "time": time_str,
                "notes": notes,
            },
        )
        return json.dumps(result)

    @function_tool
    async def reschedule_appointment(
        self, context: RunContext, booking_id: str, new_date: str, new_time: str
    ) -> str:
        """Reschedule an existing appointment.
        Args:
            booking_id: Existing booking ID
            new_date: New date in YYYY-MM-DD format
            new_time: New time in HH:MM 24hr format
        """
        result = await call_n8n(
            "reschedule",
            {"booking_id": booking_id, "new_date": new_date, "new_time": new_time},
        )
        return json.dumps(result)

    @function_tool
    async def cancel_appointment(
        self, context: RunContext, booking_id: str, reason: str = ""
    ) -> str:
        """Cancel an existing appointment.
        Args:
            booking_id: Booking ID to cancel
            reason: Reason for cancellation (optional)
        """
        result = await call_n8n(
            "cancel", {"booking_id": booking_id, "reason": reason}
        )
        return json.dumps(result)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=google_beta.realtime.RealtimeModel(
            model="gemini-3.1-flash-live-preview",
            voice="Aoede",
            temperature=0.7,
        ),
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