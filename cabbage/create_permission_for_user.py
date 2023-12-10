#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
from inspect import isclass
from typing import Any

from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import Permission, User
from django.contrib.contenttypes.models import ContentType
from django.db.models import QuerySet, Model
from django.db.models.manager import BaseManager

from spinach.models import ForecastModel, PlantModel, RecordModel, CommandModel


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


def create_user_permission(model: Any, codename: str, name: str):
    content_type = ContentType.objects.get_for_model(model)
    permission = get_query_from_db_model(Permission)
    model = qs_search(permission, codename=codename)
    if model is None:
        return permission.create(
            codename=codename,
            name=name,
            content_type=content_type,
        )

    return model


def create_row_if_not_exists(obj: Any, model: Any, **kwargs):
    query = get_query_from_db_model(obj)
    if model is None:
        query = query.create(**kwargs)
        query.save()

        return query
    return model


def main():
    """
        Permission Seeder For User!
    :return:
    """

    user = QuerySet(User)

    user = create_row_if_not_exists(
        user,
        qs_search(user, username='user'),
        username='user',
        password=make_password('user1234'),
        email='',
    )

    if isinstance(user, User):
        permission = create_user_permission(
            model=CommandModel,
            codename='can_publish_command',
            name='Can Publish Command',
        )

        user.user_permissions.add(permission)

        permission = create_user_permission(
            model=RecordModel,
            codename='can_publish_record',
            name='Can Publish Record',
        )

        user.user_permissions.add(permission)

        permission = create_user_permission(
            model=ForecastModel,
            codename='can_publish_forecast',
            name='Can Publish Forecast',
        )

        user.user_permissions.add(permission)

        permission = create_user_permission(
            model=PlantModel,
            codename='can_publish_plant',
            name='Can Publish Plant',
        )

        user.user_permissions.add(permission)

        print('spinach.can_publish_command = ', int(user.has_perm('spinach.can_publish_command')))
        print('spinach.can_publish_record = ', int(user.has_perm('spinach.can_publish_record')))
        print('spinach.can_publish_forecast = ', int(user.has_perm('spinach.can_publish_forecast')))
        print('spinach.can_publish_plant = ', int(user.has_perm('spinach.can_publish_plant')))


if str(__name__).upper() in ('__MAIN__',):
    main()
