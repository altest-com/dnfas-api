from django.apps import AppConfig


class DfapiConfig(AppConfig):
    name = 'dfapi'

    def ready(self):
        from . import signals