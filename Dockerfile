FROM python:3.5 
RUN apt-get install autoconf automake curl gcc g++ libtool pkg-config

#RUN apt-get install apt-transport-https
#RUN echo "deb https://artifacts.elastic.co/packages/5.x/apt stable main" | tee -a /etc/apt/sources.list.d/elastic-5.x.list
#RUN apt-get update && apt-get install elasticsearch
RUN wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-5.6.2.deb
RUN sha1sum elasticsearch-5.6.2.deb 
RUN dpkg -i elasticsearch-5.6.2.deb

# RUN systemctl daemon-reload
# RUN systemctl enable elasticsearch.service
# RUN systemctl start elasticsearch.service

WORKDIR /merge_machine
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN ldconfig
COPY . /merge_machine
#RUN mkdir -p merge_machine/resource/es_linker
#RUN wget "https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000@public/download/?format=json&timezone=Europe/Berlin" -O merge_machine/resource/es_linker/geonames-all-cities-with-a-population-1000.json
#RUN python3 merge_machine/es_gen_resource.py

EXPOSE 80
CMD ["uwsgi", "--http 0.0.0.0:5000 -b 32768 --wsgi-file merge_machine/api.py --callable app  --master --processes 4 --threads 2"]
CMD ["python3", "merge_machine/worker.py"]
