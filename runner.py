from schedule_meeting.graph_client import GraphClient, StaticTokenProvider
from schedule_meeting.calendar_adapter import OutlookCalendarAdapter
from schedule_meeting.models import Attendee, FindMeetingTimesRequest, TimeWindow

token_provider = StaticTokenProvider("YOUR_RAW_ACCESS_TOKEN")
graph_client = GraphClient(token_provider=token_provider)
calendar = OutlookCalendarAdapter(graph_client=graph_client)

req = FindMeetingTimesRequest(
    attendees=[
        Attendee(email="janet.zhai@ubs.com", name="Janet Zhai"),
    ],
    time_window=TimeWindow(
        start_iso="2026-03-27T09:00:00",
        end_iso="2026-03-27T18:00:00",
        timezone="Eastern Standard Time",
    ),
    duration_minutes=60,
)

result = await calendar.find_meeting_times(req)
print(result.model_dump())
