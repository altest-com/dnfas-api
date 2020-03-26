.. image:: ./images/logo.svg


.. raw:: html

    <div align="center">
      <h1>Dnfas</h1>
    </div>

|Build Status| |Test Coverage| |Python Version| |Contributions Welcome| |License|

.. |Build Status| image:: https://travis-ci.com/raikel/dnfas.svg?branch=master
   :target: https://travis-ci.com/raikel/dnfas
.. |Test Coverage| image:: https://codecov.io/gh/raikel/dnfas/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/raikel/dnfas
.. |Python Version| image:: https://img.shields.io/badge/python-v3.7+-blue.svg
   :target: http://shields.io/
.. |Contributions Welcome| image:: https://img.shields.io/badge/contributions-welcome-orange.svg
   :target: http://shields.io/
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT


A headless face analytics platform built on top of `dnfal <https://github.com/raikel/dnfal>`_ library.

Installation
=============

The following instructions are for installing Dnfas in Ubuntu 18.04 or newer. Installing on other system may work, but it has not been tested.

Prerequisites
-------------

Before you are ready to run Dnfas, you will need to install the additional software on your computer. To install them, type:

.. code-block:: bash

    sudo apt update
    sudo apt install python3-venv python3-dev libpq-dev postgresql postgresql-contrib

Python virtual environment
--------------------------

We recommend to run Dnfas within a newly created virtual environment for easier management. To create a new Python virtual environment and activate it, run the following commands within a directory of your choice:

.. code-block:: bash

    python3 -m venv pyenv
    source pyenv/bin/activate

Next, change to a directory where you wish to install Dnfas and clone the repository:

.. code-block:: bash

    git clone https://github.com/raikel/dnfas.git
    cd dnfas
    
After that, you can install python dependencies by running:

.. code-block:: bash

    pip install -r requirements.txt

Setting up the project database
-------------------------------

Next step is to create a new database and user in Postgres. We provide a small script to quickly accomplish this task. Within Dnfas project root dir, type:

.. code-block:: bash

    bash dbinit.sh \
        --pguser=${PG_USER} \
        --pgpass=${PG_PASS} \
        --dbname=${DB_NAME} \
        --dbuser=${DB_USER} \
        --dbpass=${DB_PASS} 

Where ``${PG_USER}`` is a Postgres user with super-user privileges, ``${PG_PASS}`` is the password of ``${PG_USER}``, and ``${DB_NAME}``, ``${DB_USER}`` and ``${DB_PASS}`` are the project database name, user and password, respectively. You can also set up the database by issuing the following commands in a Postgres session:

.. code-block:: bash

    DROP DATABASE IF EXISTS ${DB_NAME};
    CREATE DATABASE ${DB_NAME};
    DROP USER IF EXISTS ${DB_USER};
    CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';
    ALTER ROLE ${DB_USER} SET client_encoding TO 'utf8';
    ALTER ROLE ${DB_USER} SET default_transaction_isolation TO 'read committed';
    ALTER ROLE ${DB_USER} SET timezone TO 'UTC';
    GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};

Configuration
-------------

To configure Dnfas runtime parameters, set the following environment variables:

=======================     ===========
Variable name               Description
=======================     ===========
DJANGO_SETTINGS_MODULE      Module for Django settings. Can be "dnfas.settings.production", "dnfas.settings.development" or "dnfas.settings.testing".
DNFAS_ALLOWED_HOSTS         A comma separated list of allowed hosts, for example "192.168.5.3, 192.168.5.4".
DNFAS_SECRET_KEY            Application secret key.
DNFAS_DB_NAME               Application database name.
DNFAS_DB_USER               Application database user name.
DNFAS_DB_PASSWORD           Application database password.
DNFAS_DB_HOST               Application database host. Optional (default="localhost").
DNFAS_SPA_DIR               Root directory of Single Page Application (SPA) files. Optional (default="").
DNFAS_WORKER_NAME           Name of the current Dnfas instance when used as cluster node. Optional (default="master")
=======================     ===========
    
A configuration file with all environments variables is also provided in the project. You can find it at `deploy/dnfas.conf` under the project root directory. To use, save it to a known location and edit its content, for example:

.. code-block:: bash

    sudo cat deploy/dnfas.conf >> /etc/dnfas/dnfas.conf
    nano /etc/dnfas/dnfas.conf
    
Then set the configuration variables to appropriated values.

Application initialization
--------------------------

With configuration parameters set up and within the project virtual environment, run the following command inside the project root directory to prepare the database:

.. code-block:: bash

    python manage.py migrate
    
Next, to start the development server, type:

.. code-block:: bash

    python manage.py runserver

Serving with nginx and gunicorn
-------------------------------

Running Dnfas with the default server builtin with Django is a good way to start getting familiarized with the project. After that, however, you may want to run it with a higher performance server. Next, we describe how to setup Gunicorn and Nginx to serve Dnfas. Gunicorn will serve as an interface to Dnfas, translating client requests from HTTP to Python calls that our application can process. Nginx will be setup in front of Gunicorn to take advantage of its high performance connection handling mechanisms.

