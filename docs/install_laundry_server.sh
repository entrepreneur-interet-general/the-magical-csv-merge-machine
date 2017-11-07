# Ce fichier contient l'ensemble des commande nécessaires à une installation manuelle du backend applicatif
# sur un serveur dédié ou une machine virtuelle. L'alternative consiste à utiliser les fichiers de config Docker
# qui évitent d'avoir à exécuter ces instructions en ligne de commande.

# Prérequis : écosystème Python 

apt-get update
apt-get install python3-pip python3-dev git
pip3 install --upgrade pip

# Création d'un environnement virtuel pour isoler l'installation du backend (code Python à partir de api.py et de worker.py)

pip3 install virtualenv
mkdir ~/magical_laundry && cd ~/magical_laundry
virtualenv laundry_env
source laundry_env/bin/activate

# Installation des packages nécessaires à la compilation native d'ES, et des modules Python associés 

apt-get install autoconf automake curl gcc g++ libtool pkg-config enchant elasticsearch python3-elasticsearch

# Récupération du code source du backend applicatif

git clone https://github.com/eig-2017/the-magical-csv-merge-machine.git

# Installation du reste des modules Python requis 

cd the-magical-csv-merge-machine/
pip3 install -r requirements.txt
cd merge_machine

# Création d'une clef secrète bidon, mais nécessaire au lancement de l'API
 
echo "mmm" > secret_key.txt

# Installation d'uWSGI qui sera utilisé pour lancer le service d'API et maintenir ce dernier en fonctionnement
# (y compris en dispatchant les requêtes en frontal)

pip3 install uwsgi

# Installation + configuration d'elasticsearch

wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-5.6.2.deb /tmp
sudo dpkg -i elasticsearch-5.6.2.deb


# Préparation des données nécessaires à elasticsearch

mkdir -p resource/es_linker
wget "https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000@public/download/?format=json&timezone=Europe/Berlin" -O merge_machine/resource/es_linker/geonames-all-cities-with-a-population-1000.json
python3 es_gen_resource.py

# Ici, éditer le fichier de config /etc/default/elasticsearch et décommenter les lignes :
# LOG_DIR=/var/log/elasticsearch
# RESTART_ON_UPGRADE=true

# Ici, éditer le fichier d'options /etc/elasticsearch/jvm.options et ajouter la ligne :
# -Xms4g -Xmx4g
# (si le serveur cible a au moins 32GB de RAM, sinon réduire à 2GB, sachant qu'en-dessous ES aura du mal à construire certains index)

# Vérifier le bon fonctionnement d'elasticsearch
curl http://localhost:9200/

# Installation (compilation) de redis

wget http://download.redis.io/releases/redis-4.0.2.tar.gz
tar xzf redis-4.0.2.tar.gz
cd redis-4.0.2
make

# Lancement de redis

nohup ./src/redis-server > redis.out &

# Lancement du serveur uWSGI qui exécute le service applicatif API

uwsgi --http 0.0.0.0:5000 -b 32768 --wsgi-file api.py --callable app  --master --processes 4 --threads 2 > uwsgi.out 2>&1 &

# Lancement du worker (un seul pour l'instant : si besoin de monter en charge, envisager l'installation de composants Dockers)

nohup python3 worker.py > worker.out 2>&1 &

# Si nécessaire, ouvrir le port 5000 utilisé par l'API (rarement ouvert par défaut chez les hébergeurs)

iptables -t filter -A INPUT -p tcp --dport 5000 -j ACCEPT

# Possible à ce moment de vérifier que l'API répond sur le port 5000
