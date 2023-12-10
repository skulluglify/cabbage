#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
from inspect import isclass
from typing import Any

from django.contrib.auth.models import Permission
from django.contrib.contenttypes.models import ContentType
from django.db.models import Model, QuerySet
from django.db.models.manager import BaseManager


def is_db_model(model: Any) -> bool:
    if isclass(model):
        return issubclass(model, Model)

    return isinstance(model, Model)


def get_query_from_db_model(obj: Any) -> QuerySet:
    """
        Catch QuerySet From Any Object!
    :param obj:
    :return:
    """

    if is_db_model(obj):
        return QuerySet(obj)

    if isinstance(obj, BaseManager):
        query = obj.get_queryset()

        if isinstance(query, QuerySet):
            return query

    if isinstance(obj, QuerySet):
        return obj

    raise Exception('unable to get query set from this model')


def qs_search(model: Any, *args, **kwargs) -> Any:
    query = get_query_from_db_model(model)
    query = query.all().filter(*args, **kwargs)
    return query.first()


def create_row_if_not_exists(obj: Any, model: Any, **kwargs):
    query = get_query_from_db_model(obj)
    if model is None:
        query = query.create(**kwargs)
        query.save()

        return query
    return model


def create_user_permission(model: Any, codename: str, name: str):
    permission = Permission.objects
    content_type = ContentType.objects.get_for_model(model)
    return create_row_if_not_exists(
        permission,
        qs_search(permission, codename=codename),
        codename=codename,
        content_type=content_type,
        name=name,
    )
