import os
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

import requests
from pydantic import BaseModel
from pydantic.class_validators import root_validator
from requests.adapters import HTTPAdapter
from requests.auth import AuthBase
from slugify import slugify
from urllib3 import Retry

# NOTE: those should be on the model, but there is some pydantic issue...
DOMAIN_LEVEL_QUESTION = "Expected audience expertise: Domain"
PYTHON_LEVEL_QUESTION = "Expected audience expertise: Python"
ABSTRACT_TWEET_QUESTION = "Abstract as a tweet"
STATE_ACCEPTED = "accepted"
STATE_CONFIRMED = "confirmed"
STATE_WITHDRAWN = "withdrawn"


class Config:
    event_name: str
    pretalx_token: str
    pretalx_url: str
    base_url: str


class Speaker(BaseModel):
    code: str
    name: str
    biography: Optional[str]
    avatar: Optional[str]
    slug: str

    # Extracted
    affiliation: Optional[str] = None
    homepage: Optional[str] = None
    twitter: Optional[str] = None

    @root_validator(pre=True)
    def extract(cls, values):
        values["slug"] = slugify(values["name"])

        # This is not part of Speaker included inside the Submission, however
        # answers are included when querying the main Speaker endpoint. Because
        # we're reusing the same schema here, we can use an if to only populate
        # answers if they exist.
        if "answers" in values:
            for answer in values["answers"]:
                if cls.question_is(answer, "Company / Institute"):
                    values["affiliation"] = answer["answer"]

                if cls.question_is(answer, "Homepage"):
                    values["homepage"] = answer["answer"]

                if cls.question_is(answer, "Twitter handle"):
                    values["twitter"] = answer["answer"]

        return values

    @staticmethod
    def question_is(answer: dict, question: str) -> bool:
        return answer.get("question", {}).get("question", {}).get("en") == question


class Slot(BaseModel):
    room: str
    start: datetime
    end: datetime

    @root_validator(pre=True)
    def extract(cls, values):
        # Extracing localised data
        values["room"] = values["room"]["en"]

        return values


class Room(BaseModel):
    name: str
    # description: Optional[str]
    capacity: Optional[int]
    position: Optional[int]

    @root_validator(pre=True)
    def extract(cls, values):
        # Extracing localised data
        values["name"] = values["name"]["en"]
        if values["position"] is None:
            # If position is None then put it at the end when sorting
            values["position"] = 999

        return values


class Submission(BaseModel):
    code: str
    title: str
    speakers: List[Speaker]
    submission_type: str
    slug: str
    track: Optional[str]
    state: str
    abstract: str
    abstract_as_a_tweet: str
    description: str
    duration: str
    python_level: str = ""
    domain_level: str = ""

    # This is embedding a slot inside a submission for easier lookup later
    room: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    # Those are pre-computed down below
    talks_in_parallel: Optional[List[str]] = None
    talks_after: Optional[List[str]] = None
    next_talk_code: Optional[str] = None
    prev_talk_code: Optional[str] = None

    website_url: Optional[str] = None

    @root_validator(pre=True)
    def extract(cls, values):
        # SubmissionType and Track have localised names. For this project we
        # only care about their english versions, so we can extract them here
        for field in ["submission_type", "track"]:
            if values[field] is None:
                continue
            else:
                values[field] = values[field]["en"]

        # Some things are available as answers to questions and we can extract
        # them here
        for answer in values["answers"]:
            if cls.question_is(answer, DOMAIN_LEVEL_QUESTION):
                values["domain_level"] = answer["answer"]

            if cls.question_is(answer, PYTHON_LEVEL_QUESTION):
                values["python_level"] = answer["answer"]

            if cls.question_is(answer, ABSTRACT_TWEET_QUESTION):
                values["abstract_as_a_tweet"] = answer["answer"]

        slug = slugify(values["title"])
        values["slug"] = slug
        values["website_url"] = f"https://ep2022.europython.eu/session/{slug}"

        if values["slot"] and values["slot"]["start"] is not None:
            # NOTE: talks with multiple slots miss the slot information.
            slot = Slot.parse_obj(values["slot"])
            values["room"] = slot.room
            values["start"] = slot.start
            values["end"] = slot.end
        else:
            values["room"] = None
            values["start"] = None
            values["end"] = None

        return values

    @staticmethod
    def question_is(answer: dict, question: str) -> bool:
        return answer.get("question", {}).get("question", {}).get("en") == question

    @property
    def is_accepted(self):
        return self.state == STATE_ACCEPTED

    @property
    def is_confirmed(self):
        return self.state == STATE_CONFIRMED

    @property
    def is_publishable(self):
        return self.is_accepted or self.is_confirmed

    @property
    def is_tutorial(self):
        return "Tutorial" in self.submission_type

    @property
    def is_special_event(self):
        return "Special" in self.submission_type

    def get_talks_in_parallel(self, subs: List["Submission"]) -> Optional[List[str]]:
        if self.room is None:
            return None

        assert self.room and self.start and self.end

        output = []
        for sub in subs:
            if sub.code == self.code:
                continue

            if sub.room is None:
                continue

            assert sub.room and sub.start and sub.end

            # NOTE: should we do intersection here instead of comparison?
            if sub and sub.start == self.start:
                output.append(sub.code)

        return output

    def _set_talks_in_parallel(self, subs):
        parallel = self.get_talks_in_parallel(subs)
        self.talks_in_parallel = parallel
        return self

    def get_talks_after(self, subs: List["Submission"]) -> Optional[List[str]]:
        if self.room is None:
            return None

        assert self.room and self.start and self.end

        # Because we get timestamps from the API I'm going to simplify here and
        # assume that "talk later" is a talk that starts up to 45 minutes after
        # this talk.
        # This will *NOT* return talks that happen after a lunch break for
        # example. TBD what's the good size of the buffer
        BUFFER = timedelta(minutes=30)

        output = []
        for sub in subs:
            if sub.code == self.code:
                continue

            if sub.room is None:
                continue

            assert sub.room and sub.start and sub.end

            if sub.start > self.end and sub.start < self.end + BUFFER:
                output.append(sub.code)

        return output

    def _set_talks_after(self, subs):
        after = self.get_talks_after(subs)
        self.talks_after = after
        return self

    def get_next_talk(self, subs):
        if self.room is None:
            return None

        assert self.room and self.start and self.end

        BUFFER = timedelta(minutes=30)

        for sub in subs:
            if sub.code == self.code:
                continue

            if sub.room is None:
                continue

            assert sub.room and sub.start and sub.end

            if (
                sub.room == self.room
                and sub.start > self.end
                and sub.start < self.end + BUFFER
            ):
                return sub.code

    def _set_next_talk(self, subs):
        _next = self.get_next_talk(subs)
        self.next_talk_code = _next
        return self

    def get_prev_talk(self, subs):
        if self.room is None:
            return None

        assert self.room and self.start and self.end

        BUFFER = timedelta(minutes=50)

        for sub in subs:
            if sub.code == self.code:
                continue

            if sub.room is None:
                continue

            assert sub.room and sub.start and sub.end

            if (
                sub.room == self.room
                and sub.end < self.start
                and sub.start > self.start - BUFFER
            ):
                return sub.code

    def _set_prev_talk(self, subs):
        prev = self.get_prev_talk(subs)
        self.prev_talk_code = prev
        return self


class Pretalx:
    def __init__(self, client: Optional["PretalxClient"] = None):
        self.client = client or PretalxClient()

    def _paginate(self, url: str, limit: int = 25, offset: int = 0):
        results = []

        while 1:
            # We can't reuse the smart link from js["next"] because we have a
            # custom url concatenation on the custom client
            response = self.client.get(url, params={"limit": limit, "offset": offset})
            js = response.json()
            print("Offset %s, count %s" % (offset, js["count"]))

            results += js["results"]
            offset += limit

            if len(results) >= js["count"]:
                break

        return results

    def get_submissions(self) -> List[Submission]:
        results = self._paginate("/submissions", limit=100)
        subs = []
        # Going with a longer loop instead of list comprehension here in case
        # we need to debug validation errors from pydantic.
        for s in results:
            try:
                sub = Submission.parse_obj(s)
            except Exception:
                breakpoint()
                pass
            subs.append(sub)

        # Stable sorting
        subs = sorted(subs, key=lambda x: x.code)

        # Then fill in the scheduling details
        # TODO: instead of separate loops this should be a single loop that
        # sets all the parameters
        for sub in subs:
            sub._set_talks_in_parallel(subs)
            sub._set_talks_after(subs)
            sub._set_prev_talk(subs)
            sub._set_next_talk(subs)

        return subs

    def get_publishable_submissions(self) -> List[Submission]:
        subs = self.get_submissions()
        subs = [s for s in subs if s.is_publishable]
        return subs

    def get_speakers(self) -> Dict[str, Speaker]:
        results = self._paginate("/speakers", limit=25)
        speakers = [Speaker.parse_obj(s) for s in results]
        speakers = {s.code: s for s in speakers}
        return speakers

    def get_rooms(self):
        results = self._paginate("/rooms", limit=25)
        rooms = [Room.parse_obj(r) for r in results]
        return sorted(rooms, key=lambda x: x.position)


class PretalxError(Exception):
    pass


class PretalxClient(requests.Session):
    """ """

    base_url: str = ""

    def __init__(self, *, auth=None, base_url="", backoff_factor=1):
        """
        backoff_factor * 2 ** (number_of_failed_requests - 1)

        Values in seconds so backoff_factor=1 means retry in 0,1,2,4,8
        """
        super().__init__()

        self.backoff_factor = backoff_factor
        self.auth = auth or PretalxTokenAuth(token=Config.pretalx_token)
        self.base_url = base_url

        retry_strategy = Retry(
            total=3,  # retry three times to o a total of 4 requests
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 502],
            allowed_methods=["POST", "GET", "PUT"],
        )
        self.retry_strategy = retry_strategy
        self.mount("http://", PretalxHTTPAdapter(max_retries=retry_strategy))
        self.mount("https://", PretalxHTTPAdapter(max_retries=retry_strategy))

    def request(self, method, url, *args, **kwargs):
        url = f"{self.base_url}{url}"
        return super().request(method, url, timeout=60, *args, **kwargs)

    @classmethod
    def from_config(cls, config: Config, *args, **kwargs):
        auth = PretalxTokenAuth(token=config.pretalx_token)
        obj = cls(auth=auth, base_url=config.base_url, *args, **kwargs)
        return obj


class PretalxTokenAuth(AuthBase):
    def __init__(self, *, token):
        self.token = token

    def __call__(self, request):
        request.headers["Authorization"] = f"Token {self.token}"
        return request


class PretalxHTTPAdapter(HTTPAdapter):
    """
    Retry failed requests (with exp. backoff) and handle errors
    """

    def send(self, request, *args, **kwargs):

        response = super().send(request, *args, **kwargs)

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise PretalxError(e)

        return response


def convert_to_schedule(sessions, rooms):
    schedule = {"days": defaultdict(lambda: defaultdict(list))}

    def _according_to_room_position(x):
        return rooms.index(x)

    for s in sessions:
        if s.start is None:
            # Skip unscheduled and broken slots data
            continue

        day = s.start.date()
        day = day.strftime("%Y-%m-%d")

        if s.room not in schedule["days"][day]["rooms"]:
            schedule["days"][day]["rooms"].append(s.room)
            # This is sorting too often, but data is not big enough to worry
            # about it.
            schedule["days"][day]["rooms"].sort(key=_according_to_room_position)

        speakers = [x.name for x in s.speakers]

        def _session(start):
            return {
                "day": day,
                "ev_custom": s.title,
                "ev_duration": s.duration,
                "event_id": "",
                # NOTE: add domain level(?)
                "level": s.python_level,
                "rooms": [s.room],
                "slug": s.slug,
                "speakers": speakers,
                "start_time": start.time(),
                "talk_id": s.code,
                "time": start.time(),
                "type": s.submission_type,
                "title": s.title,
                "tt_duration": s.duration,
            }

        if s.is_tutorial:
            starts = [s.start, s.start + timedelta(minutes=90 + 15)]
        elif s.is_special_event:
            # Special events are all-day-long (6 hours) - 4 sessions, 90
            # minutes each
            starts = [
                s.start,
                s.start + timedelta(minutes=90 + 15),
                s.start + timedelta(minutes=90 + 15 + 90 + 60),
                s.start + timedelta(minutes=90 + 15 + 90 + 60 + 90 + 15),
            ]
        else:
            starts = [s.start]

        for start in starts:
            schedule["days"][day]["talks"].append(_session(start))

    return schedule


def append_breaks(schedule):
    """
    Those are hardcoded breaks, since we don't get them from the pretalx API
    """
    tutorial_days = [date(2022, 7, x).strftime("%Y-%m-%d") for x in [11, 12]]
    conference_days = [date(2022, 7, x).strftime("%Y-%m-%d") for x in [13, 14, 15]]

    def break_(name: str, start: time, duration: str):
        return {
            "day": day,
            "ev_custom": name,
            "ev_duration": duration,
            "event_id": "",
            "level": "",
            "rooms": schedule["days"][day]["rooms"],
            "slug": "",
            "speaker": "",
            "start_time": start,
            "talk_id": "",
            "time": start,
            "type": "",
            "title": name,
            "tt_duration": duration,
        }

    for day in tutorial_days:
        schedule["days"][day]["talks"].append(
            break_(
                "Coffee Break",
                start=time(11, 00),
                duration="15",
            )
        )
        schedule["days"][day]["talks"].append(
            break_(
                "Lunch Break",
                start=time(12, 30),
                duration="60",
            )
        )
        schedule["days"][day]["talks"].append(
            break_(
                "Coffee Break",
                start=time(15, 15),
                duration="15",
            )
        )

    for day in conference_days:
        schedule["days"][day]["talks"].append(
            break_(
                "Coffee Break",
                start=time(10, 00),
                duration="30",
            )
        )
        schedule["days"][day]["talks"].append(
            break_(
                "Lunch Break",
                start=time(13, 00),
                duration="60",
            )
        )
        schedule["days"][day]["talks"].append(
            break_(
                "Coffee Break",
                start=time(15, 5),
                duration="25",
            )
        )


def sort_by_start_time(schedule):
    for day in schedule["days"]:
        schedule["days"][day]["talks"] = sorted(
            schedule["days"][day]["talks"], key=lambda x: x["start_time"]
        )


def fix_duration_if_tutorial(session):
    if session.is_tutorial:
        session.duration = "180"

    return session

def fix_duration_if_special_event(session):
    # This is an all day event - four sessions 90 minutes each.
    if session.is_special_event:
        session.duration = "360"

    return session


if __name__ == "__main__":

    class Production(Config):
        event_name = "europython-2022"
        pretalx_token = os.environ["PRETALX_TOKEN"]  # THIS IS SECRET
        pretalx_url = "https://program.europython.eu"
        base_url = pretalx_url + "/api/events/" + event_name

    env = Production()
    pretalx = Pretalx(client=PretalxClient.from_config(env))

    from IPython import embed

    embed()
