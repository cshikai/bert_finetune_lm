# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvcr.io/nvidia/pytorch:19.12-py3

EXPOSE 8050

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y
RUN apt-get upgrade -y
RUN pip install --upgrade pip

# Install pip requirements
ADD requirements.txt .
RUN python -m pip install -r requirements.txt --ignore-installed
RUN pip install jupyter
# RUN python -m pip install "dask[dataframe]"
# RUN pip install fastparquet
# ADD /src /src


WORKDIR /src
# WORKDIR /src/files

#CMD ["python3", "main.py"]
