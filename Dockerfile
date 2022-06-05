# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /praca

COPY req.txt req.txt
RUN pip3 install -r req.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]