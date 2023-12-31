#!/usr/bin/env python3
from rest_framework.renderers import BaseRenderer


class PNGRenderer(BaseRenderer):
    media_type = 'image/png'
    format = 'png'
    charset = None
    render_style = 'binary'

    def render(self, data, media_type=None, renderer_context=None):
        return data
