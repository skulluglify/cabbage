#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
import json
from datetime import datetime
from typing import Dict, Any, List, Sequence, Mapping, IO

import magic
from django.contrib.auth.models import Group, Permission
from django.core.files.uploadedfile import UploadedFile
from django.core.serializers.python import Serializer
from django.db.models import Manager, QuerySet
from django.db.models.base import ModelBase, Model
from django.db.models.utils import AltersData
from django.http import HttpRequest, HttpResponse
from rest_framework.request import Request
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import RefreshToken, Token, AccessToken


def get_mime_type(buffer: IO[bytes] | UploadedFile):
    buffer.seek(0)

    m = magic.Magic(mime=True, uncompress=False)
    mime = m.from_buffer(buffer.read(16))
    buffer.seek(0)
    return mime


def merge_body_params_as_dict(request: HttpRequest | Request) -> dict:
    data = {}
    params = {}

    if isinstance(request, HttpRequest):
        data = request.POST.dict()
        params = request.GET.dict()

        if request.content_type in ('application/json',):
            body = json.loads(request.body)
            if type(body) is dict:
                data.update(body)

            data['body'] = body

    if isinstance(request, Request):
        data = dict(request.data)
        params = dict(request.query_params)

    data.update(params)
    return data


def ok_json(data: Dict[str, Any] | List[Any] | None, status=200) -> HttpResponse:
    return HttpResponse(json.dumps(data), status=status)


def get_object_from_base_model(model: ModelBase) -> Manager | QuerySet | None:
    obj = getattr(model, 'objects', None)
    if isinstance(obj, Manager) or isinstance(obj, QuerySet):
        return obj
    return None


def datetime_to_timestamp(date: datetime) -> int:
    return int(date.timestamp() * 1000)


def get(key: str, data: Any, default=None) -> Any:
    key = key.strip()

    if '.' in key:
        i = key.index('.')
        prefix = key[:i].strip()
        suffix = key[i+1:].strip()
        if prefix != '':
            if suffix != '':
                return get(suffix, get(prefix, data, default), default)
            return get(prefix, data, default)
        if suffix != '':
            return get(suffix, data, default)
        return data

    if isinstance(data, Mapping):
        return data[key] if key in data else getattr(data, key, default)

    return getattr(data, key, default)


def tokenizer(token: Token) -> Dict[str, str]:
    access_token = str(token.access_token) if isinstance(token, RefreshToken) else None
    return {
            'access': str(token) if isinstance(token, AccessToken) else access_token,
            'refresh': str(token) if isinstance(token, RefreshToken) else None,
    }


def refresh_token_verify(token: Any) -> RefreshToken | None:
    try:
        refresh_token = RefreshToken(token)
        refresh_token.verify()
        return refresh_token
    except TokenError:
        return None


class NormalizeSerializer(Serializer):
    """
        Normalize any type if is possible!\n
        - added suffix 's' for key with many values
        - hidden any key if restricted
        - table related
        - datetime
    """
    @staticmethod
    def _fields_selection_by_tag_name(tag_name: str, fields: List[str] | None = None) -> List[str] | None:
        if fields is not None:
            tag_name += '.'
            return [
                field[len(tag_name):] for field in fields
                if field.startswith(tag_name)
            ]
        return None

    @staticmethod
    def _get_tag_name_by_key(key: str) -> str | None:
        if '.' in key:
            i = key.index('.')
            key = key[:i]
            if len(key) > 0:
                return key
        return None

    def get_dump_object(self, model: Model):
        wrapper = self.__class__
        fields = self.selected_fields
        data = self._current
        restricted = type(model) in (Group, Permission)

        temp = {}
        for key in fields or data.keys():
            if key in ('password',):
                continue

            if restricted:
                if key in ('content_type',):
                    continue

            # no duplicated!
            if key in temp:
                continue

            tag_name = self._get_tag_name_by_key(key)
            tag_name_or_key = tag_name or key

            if hasattr(model, tag_name_or_key):
                field = getattr(model, tag_name_or_key)
                if isinstance(field, AltersData):

                    # model!
                    if isinstance(field, Model):
                        serializer = wrapper()
                        temp[tag_name_or_key] = serializer.serialize(
                            [field], fields=self._fields_selection_by_tag_name(tag_name_or_key, fields))
                        continue

                    # manager!
                    if hasattr(field, 'get_queryset'):
                        get_queryset = getattr(field, 'get_queryset')
                        if callable(get_queryset):
                            query = get_queryset()
                            if isinstance(query, QuerySet):
                                serializer = wrapper()
                                temp[tag_name_or_key] = serializer.serialize(
                                    query, fields=self._fields_selection_by_tag_name(tag_name_or_key, fields))
                                continue

                    temp[tag_name_or_key] = None

                elif type(field) is datetime:
                    temp[tag_name_or_key] = datetime_to_timestamp(field)

                else:
                    if isinstance(field, Sequence):
                        temp[tag_name_or_key] = field
                        continue

                    temp[tag_name_or_key] = field

            else:
                # mocked!
                if tag_name is not None:
                    temp[tag_name_or_key] = []

                temp[key] = None
        return temp
