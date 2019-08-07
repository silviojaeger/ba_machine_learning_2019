# from https://hub.docker.com/r/nvidia/cuda/

ARG cuda_version=9.0
ARG cudnn_version=7
ARG flavor=devel
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-${flavor}-ubuntu16.04

#update ubuntu stuff
RUN apt-get update

# Install Python 3.6
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get purge -y software-properties-common
RUN apt-get purge -y python3
RUN apt-get update
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip
RUN python3.6 -m pip install pip

#Install graphviz, nano and git
RUN apt-get install -y graphviz git nano

#Open Port for Tensorboard
EXPOSE 6006

#Install all required pip packages
RUN python3.6 -m pip install pandas==0.24.1
RUN python3.6 -m pip install tensorflow-gpu==1.12.2
RUN python3.6 -m pip install matplotlib==3.0.2
RUN python3.6 -m pip install Keras==2.2.4
RUN python3.6 -m pip install sklearn==0.0
RUN python3.6 -m pip install ta==0.4.5
RUN python3.6 -m pip install slackclient==1.3.1
RUN python3.6 -m pip install tables==3.5.1
