FROM python:3.7-slim
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt
ADD train.py /
ENTRYPOINT [ "python", "./train.py" ]
