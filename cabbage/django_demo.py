#!/usr/bin/env python3
from django.db.models import QuerySet

from spinach.models import RecordModel


def main():
    query = QuerySet(RecordModel)
    query = query.all()
    data = query.first()

    print(data.slots)
