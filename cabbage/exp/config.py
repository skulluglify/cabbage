#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
import json
from typing import List, Mapping, Any

import attrs


@attrs.define
class DjangoHost:
    name: str  # domain / ip!
    port: int


@attrs.define
class DjangoAppContrib:
    name: str
    includes: List[str]

    @property
    def include(self) -> str | None:
        if len(self.includes) > 0:
            return self.includes[0]

        return None


@attrs.define
class DjangoMiddleware:
    name: str
    includes: List[str]

    @property
    def include(self) -> str | None:
        if len(self.includes) > 0:
            return self.includes[0]

        return None


@attrs.define
class DjangoTemplate:
    app: bool
    backend: str
    sources: List[str]
    options: Mapping[str, Any]


@attrs.define
class DjangoDatabase:
    name: str
    options: Mapping[str, str]


@attrs.define
class DjangoValidator:
    name: str
    includes: List[str]

    @property
    def include(self) -> str | None:
        if len(self.includes) > 0:
            return self.includes[0]

        return None


@attrs.define
class DjangoStaticFile:
    name: str


@attrs.define
class DjangoRoute:
    path: str
    namespace: str | None
    includes: List[str]

    @property
    def include(self) -> str | None:
        if len(self.includes) > 0:
            return self.includes[0]

        return None


@attrs.define
class DjangoConfig:
    host: DjangoHost
    allowed_hosts: List[DjangoHost]
    apps: List[DjangoAppContrib]
    middlewares: List[DjangoMiddleware]
    templates: List[DjangoTemplate]
    databases: List[DjangoDatabase]
    validators: List[DjangoValidator]
    staticfiles: List[DjangoStaticFile]
    routes: List[DjangoRoute]


def main():
    with open('dragon.config.json', 'rb') as stream:
        data = json.load(stream)

        print(data)


if str(__name__).upper() in ('__MAIN__',):
    main()
