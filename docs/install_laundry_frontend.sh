# The fichier contient l'ensemble des commande nécessaires à une installation manuelle du frontend applicatif
# sur un serveur dédié ou une machine virtuelle. L'alternative consiste à utiliser les fichiers de config Docker
# qui évitent d'avoir à exécuter ces instructions en ligne de commande.

# Installation d'Apache et MySQL + PHPMyAdmin

apt-get install apache2 php php-curl mysql-server phpmyadmin

# Installation et activation du support PHP dans Apache
apt-get install libapache2-mod-php
a2enmod php7.0
echo "Include /etc/phpmyadmin/apache.conf" | cat >> /etc/apache2/apache2.conf


# Récupération du code source du frontend applicatif
 
git clone https://github.com/jerem1508/magical_ui.git

# Installation du source web au bon endroit
cp -r magical_ui /var/www/

# Configuration et activation du site Apache
cd /etc/apache2/sites-available
ln -s ~/magical_ui/conf/magical_ui.conf 
cd ../sites-enabled/
ln -s ../sites-available/magical_ui.conf

# Apache est maintenant bien configuré (le tester au besoin)

/etc/init.d/apache2 restart


# Config MySQL dans notre appli : ici, modifier le fichier application/config/database.php en indiquant la valeur choisie dans DB password

# Initialisation de la base MySQL

mysql --user=root --password < init.sql 

# Possible à ce moment de vérifier que l'UI répond (sur le port HTTP standard 80)