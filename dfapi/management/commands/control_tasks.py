from django.core.management.base import BaseCommand
from dfapi.services.tasks import schedule_tasks, repeat_tasks


class Command(BaseCommand):
    help = 'Start/Stop scheduled tasks'

    def handle(self, *args, **options):
        schedule_tasks()
        repeat_tasks()
        print(f'Done!')
