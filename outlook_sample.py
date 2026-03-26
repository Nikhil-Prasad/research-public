from __future__ import annotations

from typing import Any

from .graph_client import GraphClient
from .models import (
    Attendee,
    AttendeeAvailability,
    AttendeeType,
    AvailabilityStatus,
    CancelEventRequest,
    CancelEventResult,
    CandidateTime,
    CandidateTimesResult,
    CreateEventRequest,
    CreatedOrUpdatedEvent,
    FindMeetingTimesRequest,
    UpdateEventRequest,
)


class OutlookCalendarAdapter:
    def __init__(self, graph_client: GraphClient):
        self._graph = graph_client

    async def find_meeting_times(self, req: FindMeetingTimesRequest) -> CandidateTimesResult:
        payload = self._build_find_meeting_times_payload(req)
        raw = await self._graph.post("/me/findMeetingTimes", payload)
        return self._parse_find_meeting_times_response(raw)

    async def create_event(self, req: CreateEventRequest) -> CreatedOrUpdatedEvent:
        payload = self._build_create_event_payload(req)
        raw = await self._graph.post("/me/events", payload)
        return self._parse_event_response(raw)

    async def update_event(self, req: UpdateEventRequest) -> CreatedOrUpdatedEvent:
        payload = self._build_update_event_payload(req)
        raw = await self._graph.patch(f"/me/events/{req.event_id}", payload)
        return self._parse_event_response(raw)

    async def cancel_event(self, req: CancelEventRequest) -> CancelEventResult:
        payload = {"comment": req.comment or "Cancelled by schedule meeting agent."}
        await self._graph.post(f"/me/events/{req.event_id}/cancel", payload)
        return CancelEventResult(event_id=req.event_id, cancelled=True)

    async def get_event(self, event_id: str) -> CreatedOrUpdatedEvent:
        raw = await self._graph.get(f"/me/events/{event_id}")
        return self._parse_event_response(raw)

    def _build_find_meeting_times_payload(self, req: FindMeetingTimesRequest) -> dict[str, Any]:
        attendees = [self._serialize_attendee_for_graph(a) for a in req.attendees]

        payload: dict[str, Any] = {
            "attendees": attendees,
            "timeConstraint": {
                "timeslots": [
                    {
                        "start": {
                            "dateTime": req.time_window.start_iso,
                            "timeZone": req.time_window.timezone,
                        },
                        "end": {
                            "dateTime": req.time_window.end_iso,
                            "timeZone": req.time_window.timezone,
                        },
                    }
                ]
            },
            "meetingDuration": f"PT{req.duration_minutes}M",
            "returnSuggestionReasons": req.return_suggestion_reasons,
            "maxCandidates": req.max_candidates,
        }

        if req.preferred_locations:
            payload["locationConstraint"] = {
                "isRequired": req.location_required,
                "suggestLocation": True,
                "locations": [{"displayName": loc} for loc in req.preferred_locations],
            }

        return payload

    def _build_create_event_payload(self, req: CreateEventRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "subject": req.subject,
            "start": {
                "dateTime": req.start_iso,
                "timeZone": req.timezone,
            },
            "end": {
                "dateTime": req.end_iso,
                "timeZone": req.timezone,
            },
            "attendees": [self._serialize_attendee_for_event(a) for a in req.attendees],
            "allowNewTimeProposals": req.allow_new_time_proposals,
        }

        if req.body_html:
            payload["body"] = {
                "contentType": "HTML",
                "content": req.body_html,
            }

        if req.location_display_name:
            payload["location"] = {
                "displayName": req.location_display_name,
            }

        if req.is_online_meeting:
            payload["isOnlineMeeting"] = True
            payload["onlineMeetingProvider"] = req.online_meeting_provider

        return payload

    def _build_update_event_payload(self, req: UpdateEventRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        if req.subject is not None:
            payload["subject"] = req.subject

        if req.start_iso is not None and req.end_iso is not None:
            timezone = req.timezone or "UTC"
            payload["start"] = {"dateTime": req.start_iso, "timeZone": timezone}
            payload["end"] = {"dateTime": req.end_iso, "timeZone": timezone}

        if req.attendees is not None:
            payload["attendees"] = [self._serialize_attendee_for_event(a) for a in req.attendees]

        if req.body_html is not None:
            payload["body"] = {
                "contentType": "HTML",
                "content": req.body_html,
            }

        if req.location_display_name is not None:
            payload["location"] = {"displayName": req.location_display_name}

        if req.allow_new_time_proposals is not None:
            payload["allowNewTimeProposals"] = req.allow_new_time_proposals

        if req.is_online_meeting is not None:
            payload["isOnlineMeeting"] = req.is_online_meeting

        return payload

    def _parse_find_meeting_times_response(self, raw: dict[str, Any]) -> CandidateTimesResult:
        empty_reason = raw.get("emptySuggestionsReason") or None
        suggestions = raw.get("meetingTimeSuggestions", [])

        candidates: list[CandidateTime] = []
        for idx, suggestion in enumerate(suggestions, start=1):
            slot = suggestion.get("meetingTimeSlot", {})
            start = slot.get("start", {})
            end = slot.get("end", {})

            organizer_availability = self._parse_availability(
                suggestion.get("organizerAvailability")
            )

            attendee_statuses: list[AttendeeAvailability] = []
            required_ok = True

            for attendee_item in suggestion.get("attendeeAvailability", []):
                attendee = attendee_item.get("attendee", {})
                email_address = attendee.get("emailAddress", {})
                attendee_type = self._parse_attendee_type(attendee.get("type"))
                availability = self._parse_availability(attendee_item.get("availability"))

                attendee_status = AttendeeAvailability(
                    email=email_address.get("address", ""),
                    type=attendee_type,
                    availability=availability,
                )
                attendee_statuses.append(attendee_status)

                if attendee_type == AttendeeType.REQUIRED and availability != AvailabilityStatus.FREE:
                    required_ok = False

            confidence_raw = suggestion.get("confidence", 0)
            confidence = float(confidence_raw) / 100.0 if confidence_raw is not None else 0.0

            candidates.append(
                CandidateTime(
                    rank=idx,
                    start_iso=start.get("dateTime", ""),
                    end_iso=end.get("dateTime", ""),
                    timezone=start.get("timeZone", "UTC"),
                    confidence=confidence,
                    organizer_availability=organizer_availability,
                    attendee_statuses=attendee_statuses,
                    all_required_available=required_ok,
                    raw=suggestion,
                )
            )

        return CandidateTimesResult(
            candidates=candidates,
            empty_suggestions_reason=empty_reason,
        )

    def _parse_event_response(self, raw: dict[str, Any]) -> CreatedOrUpdatedEvent:
        start = raw.get("start", {})
        end = raw.get("end", {})
        online_meeting = raw.get("onlineMeeting", {}) or {}

        return CreatedOrUpdatedEvent(
            event_id=raw.get("id", ""),
            subject=raw.get("subject", ""),
            start_iso=start.get("dateTime", ""),
            end_iso=end.get("dateTime", ""),
            timezone=start.get("timeZone", "UTC"),
            web_link=raw.get("webLink"),
            join_url=online_meeting.get("joinUrl"),
            raw=raw,
        )

    def _serialize_attendee_for_graph(self, attendee: Attendee) -> dict[str, Any]:
        return {
            "emailAddress": {
                "address": attendee.email,
                "name": attendee.name or attendee.email,
            },
            "type": "Required" if attendee.type == AttendeeType.REQUIRED else "Optional",
        }

    def _serialize_attendee_for_event(self, attendee: Attendee) -> dict[str, Any]:
        return {
            "emailAddress": {
                "address": attendee.email,
                "name": attendee.name or attendee.email,
            },
            "type": "required" if attendee.type == AttendeeType.REQUIRED else "optional",
        }

    def _parse_attendee_type(self, value: str | None) -> AttendeeType:
        if (value or "").lower() == "optional":
            return AttendeeType.OPTIONAL
        return AttendeeType.REQUIRED

    def _parse_availability(self, value: str | None) -> AvailabilityStatus:
        normalized = (value or "").strip()
        for status in AvailabilityStatus:
            if status.value == normalized:
                return status
        return AvailabilityStatus.UNKNOWN
