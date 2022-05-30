import json
from datetime import datetime

from pretalx import Pretalx

pt = Pretalx()
speakers = []

SESSIONS_PATH = "./data/sessions.json"
SPEAKERS_PATH = "./data/speakers.json"


def serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()

    return obj


with open(SESSIONS_PATH, "w") as subs_fd, open(SPEAKERS_PATH, "w") as speakers_fd:
    subs = pt.get_publishable_submissions()
    js = []
    for s in subs:
        js.append(s.dict())
        speakers += [s1.dict() for s1 in s.speakers]

    subs_fd.write(json.dumps(js, indent=2, default=serialize))
    speakers_fd.write(json.dumps(speakers, indent=2, default=serialize))
