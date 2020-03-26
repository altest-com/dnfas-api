import datetime
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Thread

from django.conf import settings
from django.utils.timezone import make_aware

from ..notifications import task_notificate
from ...models import (
    Task
)

MAX_EXECUTOR_THREADS = 8
PAUSE_DURATION = 1
PROGRESS_UPDATE_INTERVAL = 5

logger_name = settings.LOGGER_NAME
logger = logging.getLogger(logger_name)


class TaskRunner(Thread):

    TASK_UPDATE_FIELDS = [
        'status',
        'updated_at',
        'started_at',
        'finished_at',
        'info',
        'progress'
    ]

    def __init__(self, task: Task, daemon: bool = True):
        super().__init__(daemon=daemon)

        self.task: Task = task
        self.status = task.status
        self.last_progress_update = 0
        self.executor = ThreadPoolExecutor(max_workers=MAX_EXECUTOR_THREADS)

    def run(self):
        try:
            self.task.status = Task.STATUS_RUNNING
            self.task.started_at = make_aware(datetime.now())
            self.send_progress()
            self.main_run()
            if self.task.status not in (
                Task.STATUS_KILLED, Task.STATUS_STOPPED
            ):
                self.task.status = Task.STATUS_SUCCESS

            self.task.finished_at = make_aware(datetime.now())
            self.send_progress()
            self.executor.shutdown(wait=True)
        except Exception as err:
            logger.error(err)
            logger.debug(traceback.format_exc())
            self.failed()
            self.executor.shutdown(wait=True)

    def pause(self):
        self.task.status = Task.STATUS_PAUSED
        self.send_progress()

    def resume(self):
        self.task.status = Task.STATUS_RUNNING
        self.send_progress()

    def stop(self):
        self.task.status = Task.STATUS_STOPPED
        self.send_progress()

    def kill(self):
        self.task.status = Task.STATUS_KILLED
        self.send_progress()

    def failed(self):
        self.task.status = Task.STATUS_FAILURE
        self.send_progress()

    def send_progress(self):
        if self.task.status != self.status:
            task_notificate(self.task.pk, self.status, self.task.status)
            self.status = self.task.status

        self.task.save(update_fields=self.TASK_UPDATE_FIELDS)
        # self.executor.submit(
        #     update_task,
        #     task=self.task,
        #     update_fields=self.TASK_UPDATE_FIELDS
        # )

    def main_run(self):
        pass
