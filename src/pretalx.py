from collections import defaultdict
from datetime import date, datetime, timedelta, time
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


class Submission(BaseModel):
    code: str
    title: str
    speakers: List[Speaker]
    submission_type: str
    slug: str
    track: str
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

        if values["slot"]:
            slot = Slot.parse_obj(values["slot"])
            values["room"] = slot.room
            values["start"] = slot.start
            values["end"] = slot.end

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

            # NOTE: should we do intersaction here instead of comparison?
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
            sub = Submission.parse_obj(s)
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


def convert_to_schedule(sessions):
    schedule = {"days": defaultdict(lambda: defaultdict(list))}

    for s in sessions:
        day = s.start.date()
        day = day.strftime("%Y-%m-%d")

        if s.room not in schedule["days"][day]["rooms"]:
            schedule["days"][day]["rooms"].append(s.room)

        speakers = [x.name for x in s.speakers]
        if speakers:
            speaker = speakers[0]
        else:
            speaker = None

        schedule["days"][day]["talks"].append(
            {
                "day": day,
                "ev_custom": s.title,
                "ev_duration": s.duration,
                "event_id": "",
                # NOTE: add domain level(?)
                "level": s.python_level,
                "rooms": [s.room],
                "slug": s.slug,
                # NOTE: there could be multiple speakers
                "speaker": speaker,
                "start_time": s.start.time(),
                "talk_id": s.code,
                "time": s.start.time(),
                "type": s.submission_type,
                "title": s.title,
                "tt_duration": s.duration,
            }
        )

    return schedule


def append_breaks(schedule):
    """
    Those are hardcoded breaks, since we don't get them from the pretalx API
    """
    for day in schedule["days"]:
        schedule["days"][day]["talks"].append({
            "day": day,
            "ev_custom": "Coffee Break",
            "ev_duration": "30",
            "event_id": "",
            "level": "",
            "rooms": schedule["days"][day]["rooms"],
            "slug": "",
            "speaker": "",
            "start_time": time(11,00),
            "talk_id": "",
            "time": time(11,00),
            "type": "",
            "title": "Coffee Break",
            "tt_duration": "30",
        })

        schedule["days"][day]["talks"].append({
            "day": day,
            "ev_custom": "Coffee Break",
            "ev_duration": "30",
            "event_id": "",
            "level": "",
            "rooms": schedule["days"][day]["rooms"],
            "slug": "",
            "speaker": "",
            "start_time": time(15,30),
            "talk_id": "",
            "time": time(15,30),
            "type": "",
            "title": "Coffee Break",
            "tt_duration": "30",
        })

        schedule["days"][day]["talks"].append({
            "day": day,
            "ev_custom": "Lunch Break",
            "ev_duration": "60",
            "event_id": "",
            "level": "",
            "rooms": schedule["days"][day]["rooms"],
            "slug": "",
            "speaker": "",
            "start_time": time(13,00),
            "talk_id": "",
            "time": time(13,00),
            "type": "",
            "title": "Lunch Break",
            "tt_duration": "60",
        })


def sort_by_start_time(schedule):
    for day in schedule["days"]:
        schedule["days"][day]["talks"] = sorted(
            schedule["days"][day]["talks"], key=lambda x: x["start_time"]
        )
