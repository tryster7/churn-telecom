FROM python:3.7
ADD preprocess.py /
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
ENTRYPOINT [ "python", "./preprocess.py" ]
