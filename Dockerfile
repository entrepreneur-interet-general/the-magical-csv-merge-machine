# Use an official Python runtime as a base image
FROM python:3.5 
#FROM continuumio/anaconda3

#RUN apk add --update curl gcc g++ \
#    && rm -rf /var/cache/apk/*
RUN apt-get install autoconf automake curl gcc g++ libtool pkg-config
#RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

RUN pip3 install bottle numpy cython --no-input 

#CMD tail -f /dev/null


# Set the working directory to /merge_machine
WORKDIR /merge_machine

# Requirements for libpostal
RUN git clone https://github.com/openvenues/libpostal
RUN cd libpostal \
	&& ./bootstrap.sh \
	&& mkdir /libpostal_data \
	&& ./configure --datadir=/libpostal_data \
	&& make \
	&& make install

RUN pip3 install numpy
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN ldconfig

# Copy the current directory contents into the container at /merge_machine
# For dev, use COPY. For prod, use github
COPY . /merge_machine

# Install requirements

# Make port XX available to the outside world
EXPOSE 80


# Define environment variables? 
WORKDIR /merge_machine/merge_machine
# Run app when container launches
CMD ["python3", "api.py"]
