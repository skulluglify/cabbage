#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'apps.DjangoMainServer.admin.settings')

application = get_asgi_application()
