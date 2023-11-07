FROM python:3.10

COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

ENTRYPOINT [ "/src/document_processing.py" ]