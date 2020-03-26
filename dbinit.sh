#!/usr/bin/env bash

SCRIPT_NAME="dbinit"

PGUSER="postgres"
PGPASS=""
DBNAME="dnfas_test"
DBUSER="dnfas_test"
DBPASS="dnfas_test"

# Print usage
usage() {
  echo -n "Usage: ${SCRIPT_NAME} [options]...

Create a new role and database for Dnfas.

Options:
  --pguser=USER     Postgres super user name (default=postgres).
  --pgpass=PASS     Postgres super user password (default='').
  --dbname=DBNAME   Name of new database to create (default=dnfas).
  --dbuser=DBUSER   User name of new role to create (default=dnfas).
  --dbpass=DBPASS   Database password (default=dnfas).
  -h, --help        Display this help and exit.

Be careful!! If a database with name DBNAME or a user with name DBUSER
already exists in Postgres, it will deleted!
"
}


for i in "$@"
do
case $i in
    -h|--help)
        usage >&2;
        exit 0;
        ;;
    --pguser=*)
        PGUSER="${i#*=}"
        shift
        ;;
    --pgpass=*)
        PGPASS="${i#*=}"
        shift
        ;;
    --dbname=*)
        DBNAME="${i#*=}"
        shift
        ;;
    --dbuser=*)
        DBUSER="${i#*=}"
        shift
        ;;
    --dbpass=*)
        DBPASS="${i#*=}"
        shift
        ;;
    *)
        echo "Invalid option: '$i'.";
        usage >&2;
        exit 1;
        ;;
esac
done


export PGPASSWORD=${PGPASS}
psql -U ${PGUSER} -h 127.0.0.1 << EOF
    DROP DATABASE IF EXISTS ${DBNAME};
    CREATE DATABASE ${DBNAME};
    DROP USER IF EXISTS ${DBUSER};
    CREATE USER ${DBUSER} WITH PASSWORD '${DBPASS}';
    ALTER ROLE ${DBUSER} SET client_encoding TO 'utf8';
    ALTER ROLE ${DBUSER} SET default_transaction_isolation TO 'read committed';
    ALTER ROLE ${DBUSER} SET timezone TO 'UTC';
    GRANT ALL PRIVILEGES ON DATABASE ${DBNAME} TO ${DBUSER};
EOF

exit 0