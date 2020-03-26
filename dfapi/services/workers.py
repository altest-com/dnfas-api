import logging
import signal
from json import JSONDecodeError
from queue import Empty as QueueEmptyError
from queue import Full as QueueFullError
from time import sleep
from typing import Dict, List

import numpy as np
import requests
import torch.multiprocessing as mp
from django import db
from django.conf import settings
from requests import Response

from .exceptions import ServiceError
from .runners import (
    TaskRunner,
    VdfTaskRunner,
    PgaTaskRunner,
    FclTaskRunner,
    VhfTaskRunner
)
from ..models import (
    Task
)

MAX_WORKERS = 4

MAX_TASKS_PER_WORKER = 8
WORKER_WAIT_TIMEOUT = 60
WORKER_PUT_TIMEOUT = 10
WORKER_QUEUE_MAX_SIZE = 24

TEST_CONN_TIMEOUT = 1

TASK_FLAG_RUN = 0
TASK_FLAG_PAUSE = 1
TASK_FLAG_RESUME = 2
TASK_FLAG_STOP = 3
TASK_FLAG_KILL = 4

logger_name = settings.LOGGER_NAME
logger = logging.getLogger(logger_name)


# def update_task(
#     task: Task,
#     update_fields: list
# ):
#     try:
#         task.save(update_fields=update_fields)
#     except Exception as err:
#         logger.exception(err)


class WorkerMessage:

    START_TASK = 'start_task'
    STOP_TASK = 'stop_task'
    KILL_TASK = 'kill_task'
    PAUSE_TASK = 'pause_task'
    RESUME_TASK = 'resume_task'

    def __init__(self, message_data: dict):
        self.task_id: int = message_data['task_id']
        self.command: str = message_data['command']


def create_task_runner(task: Task) -> TaskRunner:
    if task.task_type == Task.TYPE_VIDEO_DETECT_FACES:
        return VdfTaskRunner(task)
    elif task.task_type == Task.TYPE_VIDEO_HUNT_FACES:
        return VhfTaskRunner(task)
    elif task.task_type == Task.TYPE_PREDICT_GENDERAGE:
        return PgaTaskRunner(task)
    elif task.task_type == Task.TYPE_FACE_CLUSTERING:
        return FclTaskRunner(task)
    else:
        raise ValueError(f'Invalid task type "{task.task_type}"')


def run_worker(recv_queue: mp.Queue):

    task_runners: Dict[int, TaskRunner] = {}

    def handle_signal(_signal_number, _stack_frame):
        for runner in task_runners.values():
            runner.kill()

    signal.signal(signal.SIGQUIT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    def check_tasks():
        for key in tuple(task_runners):
            runner = task_runners[key]
            if not runner.is_alive():
                del task_runners[key]

    while True:
        try:
            message_data: dict = recv_queue.get(timeout=WORKER_WAIT_TIMEOUT)
        except QueueEmptyError:
            check_tasks()
            if len(task_runners) == 0:
                sleep(WORKER_WAIT_TIMEOUT)
                break
        else:
            message = WorkerMessage(message_data)
            command = message.command
            task_id = message.task_id

            if command == WorkerMessage.START_TASK:
                check_tasks()
                task = Task.objects.get(pk=task_id)
                task_runner = create_task_runner(task)
                task_runner.start()
                task_runners[task_id] = task_runner
            elif command == WorkerMessage.STOP_TASK:
                try:
                    task_runner = task_runners[task_id]
                except KeyError:
                    logger.error(f'Invalid task id [{task_id}].')
                else:
                    task_runner.stop()
                    del task_runners[task_id]
            elif command == WorkerMessage.KILL_TASK:
                try:
                    task_runner = task_runners[task_id]
                except KeyError:
                    logger.error(f'Invalid task id [{task_id}].')
                else:
                    task_runner.kill()
                    del task_runners[task_id]
            elif command == WorkerMessage.PAUSE_TASK:
                try:
                    task_runner = task_runners[task_id]
                except KeyError:
                    logger.error(f'Invalid task id [{task_id}].')
                else:
                    task_runner.pause()
            elif command == WorkerMessage.RESUME_TASK:
                try:
                    task_runner = task_runners[task_id]
                except KeyError:
                    logger.error(f'Invalid task id [{task_id}].')
                else:
                    task_runner.resume()
            else:
                logger.error(f'Invalid command "{command}".')


class Worker:

    def __init__(self):
        self.send_queue: mp.Queue = mp.Queue(
            maxsize=WORKER_QUEUE_MAX_SIZE
        )
        self.process: mp.Process = mp.Process(
            target=run_worker,
            kwargs={'recv_queue': self.send_queue}
        )
        self.task_ids: list = []

    def is_alive(self):
        return self.process.is_alive()

    def start(self):
        self.process.start()

    def update_index(self):
        task_ids = []
        for task_id in self.task_ids:
            try:
                task = Task.objects.get(pk=task_id)
            except Task.DoesNotExist:
                logger.error(f'Task [{task_id}] does not exists.')
            else:
                if task.status in (
                    Task.STATUS_CREATED,
                    Task.STATUS_RUNNING,
                    Task.STATUS_PAUSED
                ):
                    task_ids.append(task_id)

        self.task_ids = task_ids

    def has_task(self, task_id):
        return task_id in self.task_ids

    @property
    def task_count(self) -> int:
        return len(self.task_ids)

    def start_task(self, task_id: int):
        try:
            self.send_queue.put({
                'task_id': task_id,
                'command': WorkerMessage.START_TASK
            }, timeout=WORKER_PUT_TIMEOUT)
            self.task_ids.append(task_id)
        except QueueFullError:
            raise ServiceError(
                f'Unable to start task [{task_id}], worker queue timeout.'
            )

    def stop_task(self, task_id: int):
        if task_id not in self.task_ids:
            logger.error(f'Invalid task [{task_id}].')
            return
        try:
            self.send_queue.put({
                'task_id': task_id,
                'command': WorkerMessage.STOP_TASK
            }, timeout=WORKER_PUT_TIMEOUT)
            self.task_ids.remove(task_id)
        except QueueFullError:
            raise ServiceError(
                f'Unable to stop task [{task_id}], worker queue timeout.'
            )

    def pause_task(self, task_id: int):
        if task_id not in self.task_ids:
            logger.error(f'Invalid task [{task_id}]')
            return
        try:
            self.send_queue.put({
                'task_id': task_id,
                'command': WorkerMessage.PAUSE_TASK
            }, timeout=WORKER_PUT_TIMEOUT)
        except QueueFullError:
            raise ServiceError(
                f'Unable to pause task [{task_id}], worker queue timeout.'
            )

    def resume_task(self, task_id: int):
        if task_id not in self.task_ids:
            logger.error(f'Invalid task [{task_id}]')
            return
        try:
            self.send_queue.put({
                'task_id': task_id,
                'command': WorkerMessage.RESUME_TASK
            }, timeout=WORKER_PUT_TIMEOUT)
        except QueueFullError:
            raise ServiceError(
                f'Unable to resume task [{task_id}], worker queue timeout.'
            )


class RunnerManager:
    def __init__(self):
        self.workers: List[Worker] = []
        self.tasks_worker: Dict[int, Worker] = {}
        self._id_count = 0

    def update_index(self):
        for worker in self.workers:
            worker.update_index()

        del_ids = []
        for task_id, worker in self.tasks_worker.items():
            if not worker.has_task(task_id):
                del_ids.append(task_id)
            elif not worker.is_alive():
                del_ids.append(task_id)

        for task_id in del_ids:
            del self.tasks_worker[task_id]

        self.workers = [
            worker for worker in self.workers
            if worker.is_alive()
        ]

    def create(self, task_id: int):

        self.update_index()

        if task_id in self.tasks_worker:
            raise ServiceError(f'Task [{task_id}] is already running.')

        if len(self.workers) < MAX_WORKERS:
            worker = Worker()
            db.connections.close_all()
            worker.start()
            self.workers.append(worker)
        else:
            task_count = [worker.task_count for worker in self.workers]
            worker_ind = int(np.argmin(task_count))
            if task_count[worker_ind] >= MAX_TASKS_PER_WORKER:
                raise ServiceError(
                    f'Can not create a new task. All workers are full.'
                )
            worker = self.workers[worker_ind]

        worker.start_task(task_id)
        self.tasks_worker[task_id] = worker

    def pause(self, task_id: int):
        task_id = self.validate_task(task_id)
        self.tasks_worker[task_id].pause_task(task_id)

    def resume(self, task_id: int):
        task_id = self.validate_task(task_id)
        self.tasks_worker[task_id].resume_task(task_id)

    def stop(self, task_id: int):
        task_id = self.validate_task(task_id)
        self.tasks_worker[task_id].stop_task(task_id)
        del self.tasks_worker[task_id]

    def validate_task(self, task_id) -> int:
        try:
            task_id = int(task_id)
            if task_id not in self.tasks_worker:
                raise ValueError
            return task_id
        except ValueError:
            raise ServiceError(f'Invalid task [{task_id}].')


class WorkerApi:

    ACTION_START = 'start/'
    ACTION_PAUSE = 'pause/'
    ACTION_RESUME = 'resume/'
    ACTION_STOP = 'stop/'
    ACTION_LOGIN = 'login/'

    ACTION_CHOICES = [
        ACTION_START,
        ACTION_PAUSE,
        ACTION_RESUME,
        ACTION_STOP,
        ACTION_LOGIN
    ]

    def __init__(self, api_url: str):
        self.token = None
        self.api_url = api_url

    def _make_request(
        self,
        url: str,
        data: dict = None,
        headers: dict = None
    ):
        try:
            response = requests.post(
                url,
                data=data,
                headers=headers
            )
            if not response:
                logger.error(f'Http request error {response.status_code}.')
                self.handle_error(response)
            return response
        except requests.RequestException as err:
            logger.error(err)
            return None

    def _login(self, username: str, password: str):
        login_url = self.build_url(self.ACTION_LOGIN)
        response = self._make_request(
            login_url,
            data={
                'username': username,
                'password': password
            }
        )

        token = None
        if response is not None:
            try:
                response_data = response.json()
            except JSONDecodeError:
                pass
            else:
                if isinstance(response_data, dict):
                    token = response_data.get('token', None)

        if token is None:
            logger.warning(f'Unable to login at "{login_url}"')

        return token

    def execute(
        self,
        resource: [str, int],
        action: str,
        username: str,
        password: str
    ):
        if action not in self.ACTION_CHOICES:
            raise ValueError(f'Invalid action "{action}".')

        url = self.build_url(resource, action)

        if self.token is None and username and password:
            self.token = self._login(username=username, password=password)
            if self.token is None:
                return

        headers = None
        if self.token is not None:
            headers = {
                'Authorization': f'Token {self.token}'
            }

        self._make_request(url, headers=headers)

    @staticmethod
    def handle_error(response: Response):
        try:
            response_data = response.json()
        except JSONDecodeError:
            return

        if isinstance(response_data, list):
            for msg in response_data:
                logger.error(msg)
        elif isinstance(response_data, dict):
            for msg_key in response_data:
                logger.error(f'{msg_key}: {response_data[msg_key]}')

    def is_online(self):
        try:
            requests.head(
                self.api_url,
                timeout=TEST_CONN_TIMEOUT,
                allow_redirects=False
            )
        except:
            return False
        return True

    def build_url(self, *args):
        paths = [self.api_url] + [str(arg) for arg in args]
        return '/'.join(p.strip('/') for p in paths if p) + '/'
