from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

DESCRIPTION = (
    """
    Dnfas's back-end exposes a private API from which all system resources can 
    be accessed from any third party application.
    """
)

schema_view = get_schema_view(
    openapi.Info(
        title="Dnfas API",
        default_version='v1',
        description=DESCRIPTION,
        contact=openapi.Contact(email="raikelbl@gmail.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

swagger_view = schema_view.with_ui('swagger', cache_timeout=0)
redoc_view = schema_view.with_ui('redoc', cache_timeout=0)

