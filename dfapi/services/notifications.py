from dfapi.models import Task, Notification

TASK_OPTIONS = {
    Task.STATUS_RUNNING: {
        'title': 'Tarea iniciada',
        'message': 'Se inició la ejecución de la tarea #{}',
        'dtype': Notification.DTYPE_INFO
    },
    Task.STATUS_STOPPED: {
        'title': 'Tarea detenida',
        'message': 'La tarea #{} fue detenida manualmente',
        'dtype': Notification.DTYPE_INFO
    },
    Task.STATUS_SUCCESS: {
        'title': 'Tarea finalizada',
        'message': 'La tarea #{} finalizó correctamente',
        'dtype': Notification.DTYPE_INFO
    },
    Task.STATUS_KILLED: {
        'title': 'Tarea terminada',
        'message': 'La tarea #{} terminó de manera forzada',
        'dtype': Notification.DTYPE_WARN
    },
    Task.STATUS_FAILURE: {
        'title': 'Error de ejecución',
        'message': 'La ejecución de la tarea #{} finalizó con error',
        'dtype': Notification.DTYPE_ERROR
    }
}


def task_notificate(task_id: int, prev_status: str, next_status: str):
    if prev_status != next_status and next_status in TASK_OPTIONS.keys():
        options = TASK_OPTIONS[next_status]
        message = options['message'].format(task_id)
        Notification.objects.create(
            category=Notification.CATEGORY_TASK,
            dtype=options['dtype'],
            title=options['title'],
            message=message,
            resource=task_id,
            seen=False
        )
