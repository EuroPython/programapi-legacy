FROM python:3.10

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY Makefile .


CMD ["make", "update"]
