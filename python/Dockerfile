FROM ubuntu:bionic

RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y git python-pip

RUN pip install --upgrade pip
RUN apt-get install protobuf-compiler python-pil python-lxml

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python3", "python/app.py"]
