can you tell me what the microsoft graph API has int erms of endpoints that return a users calendar and also return their team, frequent contacts, etc? i am building a service where we are trying to schedule and manage their meetings via LLMs and am trying to figure out optimal endpoints from graph API for context engineering. 

I’ll organize the Graph API options by scheduling context—calendar/availability, meeting artifacts, org/team structure, people/contact signals, and Teams collaboration—then highlight priority endpoints, permissions, and tradeoffs for an LLM scheduler.

I’ve narrowed the important distinctions: calendarView for time-window calendar truth, getSchedule/findMeetingTimes for availability decisions, and Microsoft Search people over legacy /people for frequent-contact context.

Thought for 10m 22s

Yes. For an LLM scheduling assistant, I’d split Microsoft Graph context into four buckets: calendar truth, availability/scheduling, people/org context, and Teams/meeting artifacts. The most important distinction is that Graph has a strong calendar/free-busy model, but there is not really a single “frequent contacts” endpoint; the closest is the People API, which returns relevance-ranked people based on communication, collaboration, and business relationships.

Best endpoint stack for an LLM meeting scheduler
1. Calendar and availability
Need	Best Graph endpoint	Why it matters
User’s actual meetings in a time window	GET /me/calendarView?startDateTime=...&endDateTime=... or GET /users/{id}/calendarView?...	Best endpoint for context windows because it expands recurring meetings into concrete occurrences, including exceptions and single instances. Use this instead of plain /events for “what is on the calendar next week?”
Free/busy for one or more people or rooms	POST /me/calendar/getSchedule or POST /users/{id}/calendar/getSchedule	Gives availability blocks for users, distribution lists, rooms, or equipment. This is the deterministic endpoint I would use for candidate-slot computation.
Microsoft’s suggested meeting times	POST /me/findMeetingTimes or POST /users/{id}/findMeetingTimes	Lets Graph suggest meeting times and locations based on attendee availability and constraints. Good as a secondary candidate generator, but note that it is delegated-only and does not support application permissions.
User preferences	GET /me/mailboxSettings/timeZone, GET /me/mailboxSettings/workingHours, or GET /me/mailboxSettings	Essential for prompt/context engineering: timezone, working hours, automatic replies, locale, and related mailbox settings.
Rooms and room lists	GET /places/microsoft.graph.room, GET /places/microsoft.graph.roomlist, GET /places/{room-list-email}/microsoft.graph.roomlist/rooms	Use for room-aware scheduling, hybrid meetings, location constraints, capacity, and room metadata.
Create a calendar event	POST /me/calendar/events or POST /users/{id}/calendar/events	Use after the LLM proposes and the user confirms. For Teams meetings, create an event with online meeting fields rather than creating a standalone online meeting first.
Keep local cache fresh	GET /me/calendarView/delta?... plus Graph change notifications	Use delta for incremental sync and subscriptions/webhooks to avoid constant polling.

For your use case, the default real-time flow should be:

1. Resolve attendees.
2. Pull user mailbox settings.
3. Pull the user’s calendarView for the relevant window.
4. Pull attendee and room free/busy via getSchedule.
5. Optionally call findMeetingTimes if running delegated.
6. Let the LLM rank/explain candidate slots.
7. Create or update the event via calendar events API.
8. Keep future state fresh with delta + subscriptions.
2. People, contacts, org structure, and “team”

There are three different meanings of “team” in Graph: Microsoft Teams membership, organizational hierarchy, and collaboration/frequent-people signals.

Need	Endpoint	Notes
Relevant / frequent collaborators	GET /me/people or GET /users/{id}/people	Closest endpoint to “frequent contacts.” It returns person objects ordered by relevance from communication/collaboration patterns and business relationships. Supports $search, $select, $top, etc.
Saved Outlook contacts	GET /me/contacts or GET /users/{id}/contacts	This is the user’s actual contacts folder, not necessarily frequent collaborators. Useful for external contacts and manually saved people.
User profile / directory profile	GET /me, GET /users/{id}, GET /users/{id}?$select=displayName,mail,jobTitle,department,officeLocation,userPrincipalName	Use for canonical person metadata once you resolve a person. Default user fields include display name, mail, job title, office location, UPN, phones, and related fields.
Manager	GET /me/manager or GET /users/{id}/manager	Useful for escalation, priority heuristics, and “my manager asked…” scheduling logic.
Direct reports	GET /me/directReports or GET /users/{id}/directReports	Useful for team meetings, skip-level meetings, and org-context ranking.
Groups / distribution lists / M365 groups	GET /me/memberOf, GET /me/transitiveMemberOf	Useful for inferring department, project groups, access groups, and possible team aliases. transitiveMemberOf includes direct and indirect memberships.
Microsoft Teams the user belongs to	GET /me/joinedTeams or GET /users/{id}/joinedTeams	Returns the Teams where the user is a direct member. Basic team fields only; call “get team” for richer team details.
Teams including shared-channel association	GET /users/{user-id}/teamwork/associatedTeams	Use when shared channels matter. This can return teams associated through shared channel membership, not only direct team membership.
Members of a Team	GET /teams/{team-id}/members	Useful for “schedule with everyone in my project team,” after resolving the team.

For “frequent contacts,” I would start with:

GET /me/people?$top=50&$select=displayName,userPrincipalName,scoredEmailAddresses,jobTitle,department,personType

Then enrich selected people with:

GET /users/{id-or-upn}?$select=displayName,mail,userPrincipalName,jobTitle,department,officeLocation
GET /users/{id-or-upn}/manager

For “my team,” do not rely on just one endpoint. I’d combine:

GET /me/manager
GET /me/directReports
GET /me/joinedTeams
GET /me/memberOf

Then build your own internal notion of:

manager
direct_reports
close_collaborators
teams_memberships
groups_memberships
3. Teams meeting artifacts

If you are managing meetings, you may also want Teams-specific data, but I would treat this as an advanced layer because permissions get sensitive fast.

Need	Endpoint	Notes
Create a Teams meeting	Create an Outlook event with isOnlineMeeting: true and onlineMeetingProvider: "teamsForBusiness"	Best path for calendar-native scheduling. The event resource exposes online meeting details such as join information.
Get online meeting details	GET /me/onlineMeetings/{meetingId} or GET /me/onlineMeetings?$filter=JoinWebUrl eq '{joinWebUrl}'	Useful when mapping a calendar event’s Teams join URL back to an online meeting object. Application access requires additional policy setup.
Transcripts	GET /me/onlineMeetings/{id}/transcripts or /users/{user-id}/onlineMeetings/{id}/transcripts	Useful for post-meeting summaries, but requires high-sensitivity transcript permissions and only applies to supported scheduled online meetings.
Recordings	GET /me/onlineMeetings/{id}/recordings	Similar caveats as transcripts. Use only if you truly need post-meeting artifact processing.
Attendance reports	GET /me/onlineMeetings/{id}/attendanceReports	Useful for “who attended?” and follow-up automation.
User chats	GET /me/chats	Potentially useful for context, but I would avoid by default. It is much more privacy-sensitive than calendar/people metadata.
Chat messages / channel messages	GET /chats/{chat-id}/messages, GET /teams/{team-id}/channels/{channel-id}/messages	Only use with explicit product justification, narrow scopes, and strong privacy controls.

For most scheduling assistants, I would not include Teams chat messages in the first version. Calendar, free/busy, people relevance, org hierarchy, Teams memberships, and meeting artifacts are usually enough.

Recommended context schema for the LLM

You generally do not want to dump raw Graph objects into the prompt. Normalize them first:

{
  "user": {
    "displayName": "Nikhil",
    "timeZone": "America/New_York",
    "workingHours": {
      "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
      "start": "09:00",
      "end": "17:00"
    }
  },
  "request": {
    "intent": "schedule_meeting",
    "durationMinutes": 30,
    "requestedWindow": {
      "start": "2026-06-03T09:00:00-04:00",
      "end": "2026-06-07T17:00:00-04:00"
    },
    "attendees": ["person-a@company.com", "person-b@company.com"]
  },
  "availability": [
    {
      "principal": "person-a@company.com",
      "busy": [
        {
          "start": "2026-06-03T10:00:00-04:00",
          "end": "2026-06-03T10:30:00-04:00",
          "status": "busy"
        }
      ]
    }
  ],
  "people_context": [
    {
      "email": "person-a@company.com",
      "displayName": "Person A",
      "relationshipSignal": "relevant_person",
      "jobTitle": "Engineering Manager",
      "department": "AI Platform",
      "isManager": false,
      "isDirectReport": false,
      "sharedTeams": ["AI Platform"]
    }
  ],
  "constraints": {
    "avoidPrivateEventBodies": true,
    "respectWorkingHours": true,
    "minimumAttendeePercentage": 100,
    "allowOverbooking": false,
    "preferTeamsMeeting": true
  }
}

The LLM should reason over availability blocks, relationship signals, and constraints, not raw emails, raw event bodies, or full chat histories.

Suggested MVP permissions

For a delegated assistant acting on behalf of the signed-in user:

User.Read
Calendars.ReadWrite
MailboxSettings.Read
People.Read
Contacts.Read          # optional
User.ReadBasic.All     # for basic org/person lookup
Team.ReadBasic.All     # for joined Teams
Place.Read.All         # for rooms

Add these only when truly needed:

User.Read.All
Group.Read.All
OnlineMeetings.Read
OnlineMeetings.ReadWrite
OnlineMeetingTranscript.Read.All
OnlineMeetingRecording.Read.All
OnlineMeetingArtifact.Read.All
Chat.Read
ChannelMessage.Read.All

Delegated permissions mean the app acts on behalf of a signed-in user and cannot go beyond that user’s access; application permissions let the app act as itself and may access data broadly, usually requiring admin consent. Microsoft recommends least privilege.

For an enterprise app using application permissions against Exchange calendar/contact/mailbox data, scope mailbox access carefully. Microsoft now recommends Exchange RBAC for Applications for scoping app access to Exchange Online resources; older Application Access Policies are described as legacy and replaced by App RBAC for new configurations.

My recommended production design

Use calendarView, mailboxSettings, people, org endpoints, and getSchedule as the primary context layer. Use findMeetingTimes only when you are in a delegated flow and want Microsoft’s candidate suggestions. For application-only or backend scheduling, compute candidate slots yourself from getSchedule.

A strong implementation would look like this:

Real-time per request:
- /me/mailboxSettings
- /me/calendarView
- /me/people with search/top/select
- /users/{id}/manager or /directReports when needed
- /me/joinedTeams when team context matters
- /me/calendar/getSchedule for candidate attendees and rooms

Cached daily or weekly:
- /me/memberOf or /me/transitiveMemberOf
- /me/joinedTeams or /associatedTeams
- /teams/{team-id}/members
- /places/microsoft.graph.room and room lists
- enriched user profiles

Sync layer:
- calendarView delta
- change notifications for calendar events
- optional contacts delta/change notifications

For your LLM context engineering, the most valuable endpoints are probably:

GET  /me/calendarView?startDateTime=...&endDateTime=...
POST /me/calendar/getSchedule
GET  /me/mailboxSettings
GET  /me/people
GET  /me/manager
GET  /me/directReports
GET  /me/joinedTeams
GET  /me/memberOf
GET  /places/microsoft.graph.room
POST /me/calendar/events
PATCH /me/events/{id}

That gives you enough to schedule intelligently without overreaching into highly sensitive surfaces like Teams chat content.
