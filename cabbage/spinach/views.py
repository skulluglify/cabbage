import io
import os.path
from secrets import compare_digest
from typing import Any, Dict, List

from django.contrib.auth.models import User, Group
from django.core.files.uploadedfile import UploadedFile
from django.db.models import QuerySet
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.utils.datastructures import MultiValueDict
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.vary import vary_on_headers
from rest_framework import viewsets, permissions, views, status
from rest_framework.decorators import renderer_classes
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.tokens import RefreshToken

from . import settings
from .base import Record, Command
from .models import RecordModel, CommandModel, ProfileModel, ForecastModel
from .odac_plant_disease import odac_plant_disease_predictor
from .renderers import PNGRenderer
from .serializers import UserSerializer, GroupSerializer, MyTokenObtainPairSerializer
from .utils import (merge_body_params_as_dict, NormalizeSerializer, tokenizer, get, refresh_token_verify,
                    get_mime_type)


class LoginView(views.APIView):
    renderer_classes = [TemplateHTMLRenderer]

    @staticmethod
    def get(request: Request) -> Response:
        context = {
            'meta': {
                'name': 'Login',
            }
        }

        return Response(
            data=context,
            template_name='spinach/login.html',
            content_type='text/html')


class LogoutView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    @method_decorator(vary_on_headers("Authorization"))
    def post(self, request: Request) -> Response:

        jwt = JWTAuthentication()
        auth = jwt.authenticate(request)

        refresh_token = get('token.refresh', request.data)
        if refresh_token is None:
            return Response({
                'message': 'require token.refresh for process',
            }, status=status.HTTP_400_BAD_REQUEST)

        refresh_token = refresh_token_verify(refresh_token)

        if auth is not None:
            auth_user, token = auth

            # TODO: fix token outstanding
            if auth_user.username == get('payload.username', refresh_token):
                if not isinstance(token, RefreshToken):
                    token = refresh_token
                token.blacklist()

                return Response({
                    'message': 'token deleted',
                }, status=status.HTTP_200_OK)

            return Response({
                'message': 'refresh token not valid',
            }, status=status.HTTP_400_BAD_REQUEST)

        return Response({
            'message': 'access denied',
        }, status=status.HTTP_401_UNAUTHORIZED)


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all().order_by('-date_joined')  # desc
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]


class ProfileView(views.APIView):

    @method_decorator(vary_on_headers('Authorization'))
    def get(self, request: Request) -> Response:

        jwt = JWTAuthentication()
        auth = jwt.authenticate(request)

        if auth is not None:
            auth_user, token = auth
            username = auth_user.username

            if username is not None:
                query = QuerySet(ProfileModel)
                query = query.all()
                query = query.select_related('user')
                query = query.filter(user=auth_user, deleted_at=None)

                serializer = NormalizeSerializer()
                result = serializer.serialize(query, fields=[
                    'user.username',
                    'user.email',
                    'user.last_login',
                    'user.is_superuser',
                    'user.is_staff',
                    'user.is_activate',
                    'user.date_joined',
                    'user.user_permissions.name',
                    'user.user_permissions.codename',
                    'nickname',
                    'gender',
                    'website',
                    'picture',
                    'created_at',
                    'updated_at',
                ])

                return Response({
                    'message': 'successful get user data',
                    'data': result,
                }, status=status.HTTP_200_OK)

            return Response({
                'message': 'unable to get user data',
                'data': None,
            }, status=status.HTTP_400_BAD_REQUEST)

        return Response({
            'message': 'access denied',
        }, status=status.HTTP_401_UNAUTHORIZED)

    @method_decorator(csrf_protect)
    def post(self, request: Request) -> Response:
        username = get('username', request.data)
        password = get('password', request.data)

        if username is not None:
            if password is not None:
                query = QuerySet(User)
                query = query.all()
                query = query.filter(username=username)
                user = query.first()

                if isinstance(user, User):
                    if user.check_password(password):
                        token = MyTokenObtainPairSerializer.get_token(user)

                        return Response({
                            'message': 'login successfully',
                            # 'secrets': {
                            #     'IOT_HYDROPONIC_PROJECT_SECRET': settings.IOT_HYDROPONIC_PROJECT_SECRET,
                            #     'ODAC_PLANT_DISEASE_SECRET': settings.ODAC_PLANT_DISEASE_SECRET,
                            # },
                            'token': tokenizer(token),
                        }, status=status.HTTP_200_OK)

        return Response({
            'message': 'login failed',
        }, status=status.HTTP_401_UNAUTHORIZED)


class ODACPlantDiseaseView(views.APIView):
    @odac_plant_disease_predictor
    def predict(self, request: Request, predictions: List[Dict[str, Any]] | None = None):
        if predictions is None:
            predictions = []

        scheme = get('scheme', request, 'http')
        host = scheme + '://' + get('META.HTTP_HOST', request, 'localhost')

        token = get('token', request.query_params, '')
        query_params = '?token=' + token

        for idx, prediction in enumerate(predictions):
            image_data_boxes = prediction['image_data_boxes']
            image_data_boxes = host + '/odac/plant-disease/images/' + image_data_boxes + query_params
            prediction['image_data_boxes'] = image_data_boxes
            predictions[idx] = prediction

            # saving into database!
            query = QuerySet(ForecastModel)
            query = query.create(image=image_data_boxes, result=prediction)
            query.save()

        return predictions

    # @method_decorator(csrf_protect)
    @method_decorator(vary_on_headers('Content-Disposition', 'Content-Type', 'Content-Length'))
    def post(self, request: Request) -> Response:
        token = get('token', request.query_params, '')
        files = request.FILES

        if not compare_digest(token, settings.ODAC_PLANT_DISEASE_SECRET):
            return Response({
                'message': 'access denied',
                'data': [],
            }, status=status.HTTP_401_UNAUTHORIZED)

        if isinstance(files, MultiValueDict):
            image_sample = files.get('image_sample')
            image_sample_content_types = ('image/jpeg', 'image/png')

            if isinstance(image_sample, UploadedFile):
                if image_sample.content_type in image_sample_content_types:
                    mime = get_mime_type(image_sample)

                    if mime in image_sample_content_types:
                        buffer = io.BytesIO()
                        buffer.seek(0)
                        buffer.truncate(0)

                        for chunk in image_sample.chunks():
                            buffer.write(chunk)

                        # get data prediction!
                        predictions = self.predict(request, fp=buffer)

                        return Response({
                            'message': 'successful predict plant disease',
                            'data': predictions,
                        })

                return Response({
                    'message': 'bad mime type',
                }, status=status.HTTP_400_BAD_REQUEST)

        return Response({
            'message': 'multipart-form data image_sample is not found',
        }, status=status.HTTP_400_BAD_REQUEST)


class ODACPlantDiseaseImagesView(views.APIView):
    renderer_classes = [PNGRenderer]

    @method_decorator(vary_on_headers('Host'))
    def get(self, request: Request, *args, **kwargs) -> Response:
        token = get('token', request.query_params, '')

        dirname, basename = os.path.split(kwargs.get('path'))
        dirname, basename = dirname.strip(), basename.strip()
        path = os.path.join(settings.WORKDIR, '../data/odac/plant/prediction/outputs', basename)
        if dirname != '' or basename == '':
            # return Response({
            #     'message': 'what you want? get out!',
            #     'data': [],
            # }, status=status.HTTP_401_UNAUTHORIZED)
            return Response(b'', status=status.HTTP_401_UNAUTHORIZED)

        if not compare_digest(token, settings.ODAC_PLANT_DISEASE_SECRET):
            # return Response({
            #     'message': 'access denied',
            #     'data': [],
            # }, status=status.HTTP_401_UNAUTHORIZED)
            return Response(b'', status=status.HTTP_401_UNAUTHORIZED)

        if os.path.exists(path):
            if os.path.isfile(path):
                with open(path, 'rb') as stream:
                    data = stream.read()
                    return Response(data, content_type='image/png')

        # return Response({
        #     'message': 'multipart-form data image_sample is not found',
        # }, status=status.HTTP_400_BAD_REQUEST)
        return Response(b'', status=status.HTTP_400_BAD_REQUEST)


class ForecastImagesView(views.APIView):

    @renderer_classes([PNGRenderer])
    @method_decorator(vary_on_headers('Host'))
    def get(self, request: Request, *args, **kwargs) -> Response:
        token = get('token', request.query_params, '')

        if not compare_digest(token, settings.ODAC_PLANT_DISEASE_SECRET):
            return Response({
                'message': 'access denied',
                'data': [],
            }, status=status.HTTP_401_UNAUTHORIZED)

        query = QuerySet(ForecastModel)
        query = query.all()
        query = query.order_by('-created_at')

        data = []
        forecasts = query[:]
        for forecast in forecasts:
            if isinstance(forecast, ForecastModel):
                data.append(forecast.result)

        return Response({
            'message': 'successful get data predictions',
            'data': data,
        }, status=status.HTTP_200_OK)


def index(request: HttpRequest) -> HttpResponse:
    return render(request, template_name='spinach/index.html',
                  context={
                      'meta': {
                          'name': 'Spinach',
                      }
                  })


class RecordView(views.APIView):

    @method_decorator(vary_on_headers('Host'))
    def get(self, request: Request, *args, **kwargs) -> Response:
        query = QuerySet(RecordModel)
        query = query.all()
        query = query.filter(dummy=False)
        query = query.order_by('-created_at')
        serializer = NormalizeSerializer()
        data = serializer.serialize(query, fields=[
            'UUID',
            'ph',
            'ph_fuzzy',
            'ec',
            'air_temp',
            'humidity',
            'water_flow',
            'lighting',
            'full_water_tank',
            'acid_actuator',
            'alkaline_actuator',
            'nutrient_actuator',
            'fans_rpm',
            'created_at',
        ])

        return Response({
            'message': 'successful get data records',
            'data': data,
        })

    @method_decorator(vary_on_headers('Host'))
    def post(self, request: Request, *args, **kwargs) -> Response:
        token = get('token', request.query_params, '')

        if not compare_digest(token, settings.IOT_HYDROPONIC_PROJECT_SECRET):
            return Response({
                'message': 'access denied',
                'data': [],
            }, status=status.HTTP_401_UNAUTHORIZED)

        columns = [
            "ph", "ph_fuzzy", "ec", "air_temp", "humidity", "water_flow", "lighting", "full_water_tank",
            "acid_actuator", "alkaline_actuator", "nutrient_actuator", "fans_rpm",
        ]

        data = merge_body_params_as_dict(request)
        merge = get('merge', data, ','.join(columns))  # key 'merge' not found, update all!
        merge_columns = list(map(lambda x: str(x).strip().lower(), str(merge).strip().split(',')))
        merge_columns = list(filter(lambda x: x is not None and x != '', merge_columns))
        merge_columns = [column for column in merge_columns if column in columns]

        UUID = get('UUID', data, get('uuid', data))

        record = Record(**data)

        query = QuerySet(RecordModel)
        query = query.all()

        if UUID is not None:
            query = query.filter(UUID=UUID)

        else:
            query = query.filter(dummy=True)

        query = query.order_by('-created_at')
        check = query.first()

        # create new dummy first time!
        if check is None:
            if UUID is not None:
                return Response({
                    'message': f'record with uuid {UUID} is not found',
                }, status=status.HTTP_400_BAD_REQUEST)

            record = Record(
                ph=0.0,
                ph_fuzzy=0.0,
                ec=0.0,
                air_temp=0.0,
                humidity=0.0,
                water_flow=0.0,
                lighting=False,
                full_water_tank=False,
                acid_actuator=False,
                alkaline_actuator=False,
                nutrient_actuator=False,
                fans_rpm=0,
            )

            query = QuerySet(RecordModel)
            query = query.create(**record.dict())
            query.save()

            check = query

        if isinstance(check, RecordModel):
            slots = check.slots or ''
            slots_columns = list(map(lambda x: str(x).strip().lower(), str(slots).strip().split(',')))
            slots_columns = list(filter(lambda x: x is not None and x != '', slots_columns))
            slots_columns = list(set(merge_columns) | set(slots_columns))

            # append new value in the current data record!
            for column in merge_columns:
                value = get(column, record, None)
                if value is not None:
                    setattr(check, column, value)

            check.slots = ','.join(slots_columns)
            check.save()

            diff_columns = set(columns) ^ set(slots_columns)

            # create new dummy after full task complete!
            if len(diff_columns) == 0:
                check.dummy = False
                check.save()

                record = Record(
                    ph=0.0,
                    ph_fuzzy=0.0,
                    ec=0.0,
                    air_temp=0.0,
                    humidity=0.0,
                    water_flow=0.0,
                    lighting=False,
                    full_water_tank=False,
                    acid_actuator=False,
                    alkaline_actuator=False,
                    nutrient_actuator=False,
                    fans_rpm=0,
                )

                query = QuerySet(RecordModel)
                query = query.create(**record.dict())
                query.save()

                check = query

            return Response({
                'message': 'successful save records',
                'data': {
                    'UUID': check.UUID,
                },
            })

        return Response({
            'message': 'failed create data record',
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CommandView(views.APIView):

    @method_decorator(vary_on_headers('Host'))
    def get(self, request: Request, *args, **kwargs) -> Response:
        size = int(get('size', request.query_params, '1'))
        query = QuerySet(CommandModel)
        query = query.all()
        query = query.order_by('-created_at')
        query = query[:size]
        serializer = NormalizeSerializer()
        data = serializer.serialize(query)

        return Response({
            'message': 'successful get data commands',
            'data': data,
        })

    @method_decorator(vary_on_headers('Host'))
    def post(self, request: Request, *args, **kwargs) -> Response:
        token = get('token', request.query_params, '')

        if not compare_digest(token, settings.IOT_HYDROPONIC_PROJECT_SECRET):
            return Response({
                'message': 'access denied',
                'data': [],
            }, status=status.HTTP_401_UNAUTHORIZED)

        data = merge_body_params_as_dict(request)
        command = Command(**data)

        query = QuerySet(CommandModel)
        query = query.create(**command.dict())
        query.save()

        return Response({
            'message': 'successful save commands',
        })
