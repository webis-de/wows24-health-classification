FROM python:3.10

COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

ENTRYPOINT [ "/workspace/src/processing.py" ]