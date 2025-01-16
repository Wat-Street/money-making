# THIS IS A TEMPLATE
# If you're new to Docker and you're interested in what the commands do, the Docker reference is below:
#    https://docs.docker.com/reference/dockerfile/
# If you're confused on how to fill this template out, refer to the sample model directory located 
# in `money-making/projects/orderbook_test_model`.
FROM python:3.10-slim

WORKDIR ./

# copy necessary files
COPY <path_to_model_file> .
COPY <path_to_requirements.txt> .

# install dependencies
RUN pip install -r requirements.txt

# a command that does nothing, but keeps the container running indefinitely until we kill it.
CMD ["sleep", "infinity"]
