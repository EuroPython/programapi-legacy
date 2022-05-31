import json
import os
from datetime import datetime

from pretalx import Config, Pretalx, PretalxClient


class Staging(Config):
    event_name = "staging-europython-2022"
    pretalx_token = os.environ["PRETALX_TOKEN"]  # THIS IS SECRET
    pretalx_url = "https://program.europython.eu"
    base_url = pretalx_url + "/api/events/" + event_name

    SESSIONS_PATH = "./data/staging-sessions.json"
    SPEAKERS_PATH = "./data/staging-speakers.json"


class Production(Config):
    event_name = "europython-2022"
    pretalx_token = os.environ["PRETALX_TOKEN"]  # THIS IS SECRET
    pretalx_url = "https://program.europython.eu"
    base_url = pretalx_url + "/api/events/" + event_name

    SESSIONS_PATH = "./data/sessions.json"
    SPEAKERS_PATH = "./data/speakers.json"


def serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()

    return obj


for env in Staging(), Production():
    pretalx = Pretalx(client=PretalxClient.from_config(env))

    speakers = []
    with open(env.SESSIONS_PATH, "w") as sessions_fd, open(
        env.SPEAKERS_PATH, "w"
    ) as speakers_fd:
        subs = pretalx.get_publishable_submissions()
        extra_speakers_info = pretalx.get_speakers()

        sessions = []
        for s in subs:
            # Backfill all the data from the other endpoint
            # To simplify just overwrite full list of objects
            s.speakers = [extra_speakers_info[s.code] for s in s.speakers]
            sessions.append(s.dict())
            speakers += [s1.dict() for s1 in s.speakers]

        sessions_fd.write(json.dumps(sessions, indent=2, default=serialize))
        speakers_fd.write(json.dumps(speakers, indent=2, default=serialize))
