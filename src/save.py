import json
import os
from datetime import datetime

import requests

from pretalx import (
    Config,
    Pretalx,
    PretalxClient,
    append_breaks,
    convert_to_schedule,
    fix_duration_if_special_event,
    fix_duration_if_tutorial,
    sort_by_start_time,
)


class Staging(Config):
    event_name = "staging-europython-2022"
    pretalx_token = os.environ["PRETALX_TOKEN"]  # THIS IS SECRET
    pretalx_url = "https://program.europython.eu"
    base_url = pretalx_url + "/api/events/" + event_name

    SESSIONS_PATH = "./data/staging-sessions.json"
    SPEAKERS_PATH = "./data/staging-speakers.json"
    SCHEDULE_PATH = "./data/staging-schedule.json"


class Production(Config):
    event_name = "europython-2022"
    pretalx_token = os.environ["PRETALX_TOKEN"]  # THIS IS SECRET
    pretalx_url = "https://program.europython.eu"
    base_url = pretalx_url + "/api/events/" + event_name
    publich_schedule_url = (
        pretalx_url + "/europython-2022/schedule/export/schedule.json"
    )

    SESSIONS_PATH = "./data/sessions.json"
    SPEAKERS_PATH = "./data/speakers.json"
    SCHEDULE_PATH = "./data/schedule.json"


def serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()

    return str(obj)


for env in Staging(), Production():
    pretalx = Pretalx(client=PretalxClient.from_config(env))

    speakers = []
    subs = pretalx.get_publishable_submissions()
    extra_speakers_info = pretalx.get_speakers()
    rooms = [r.name for r in pretalx.get_rooms()]

    # At this point all tutorials are still 90 minutes in the responses,
    # because that's how they are configured in pretalx.
    schedule = convert_to_schedule(subs, rooms)
    schedule["generated_at"] = datetime.utcnow()

    sessions = []
    for s in subs:
        # Backfill all the data from the other endpoint
        # To simplify just overwrite full list of objects
        s.speakers = [extra_speakers_info[s.code] for s in s.speakers]
        s = fix_duration_if_tutorial(s)
        s = fix_duration_if_special_event(s)
        sessions.append(s.dict())
        speakers += [s1.dict() for s1 in s.speakers]

    if "staging" not in env.event_name:
        # Breaks are hardcoded and work only for production schedule at the
        # moment.
        append_breaks(schedule)
        sort_by_start_time(schedule)

    # Open files at the very end so that in case of API failure we don't
    # truncate the file.
    with open(env.SESSIONS_PATH, "w") as sessions_fd, open(
        env.SPEAKERS_PATH, "w"
    ) as speakers_fd, open(env.SCHEDULE_PATH, "w") as schedule_fd:
        sessions_fd.write(json.dumps(sessions, indent=2, default=serialize))
        speakers_fd.write(json.dumps(speakers, indent=2, default=serialize))
        schedule_fd.write(json.dumps(schedule, indent=2, default=serialize))
