import sys
from fabric import Connection
from invoke import task

PROJECT_NAME = "project_name"
PROJECT_PATH = "~/{}".format(PROJECT_NAME)
VENV_PATH = '/home/ronin/Projects/pyenvs/bin'
REPO_URL = 'https://github.com/raikel/dnfas.git'


@task
def upgrade(c):
    c.run(f'git pull')
    c.run(f'{VENV_PATH}/pip install -r requirements.txt')
    c.run(f'{VENV_PATH}/python manage.py collectstatic --noinput')
    c.run(f'{VENV_PATH}/python manage.py migrate --noinput')




# supervisor tasks
@task
def start(ctx):
    if isinstance(ctx, Connection):
        conn = ctx
    else:
        conn = get_connection(ctx)
    conn.sudo("supervisorctl start all")


@task
def restart(ctx):
    if isinstance(ctx, Connection):
        conn = ctx
    else:
        conn = get_connection(ctx)
    print("restarting supervisor...")
    conn.sudo("supervisorctl restart all")

@task
def stop(ctx):
    if isinstance(ctx, Connection):
        conn = ctx
    else:
        conn = get_connection(ctx)
    conn.sudo("supervisorctl stop all")


@task
def status(ctx):
    if isinstance(ctx, Connection):
        conn = ctx
    else:
        conn = get_connection(ctx)
    conn.sudo("supervisorctl status")


# deploy task
@task
def deploy(ctx):
    conn = get_connection(ctx)
    if conn is None:
        sys.exit("Failed to get connection")
    clone(conn)
    with conn.cd(PROJECT_PATH):
        print("checkout to dev branch...")
        checkout(conn, branch="dev")
        print("pulling latest code from dev branch...")
        pull(conn)
        print("migrating database....")
        migrate(conn)
        print("restarting the supervisor...")
        restart(conn)