from django.urls import include, path
from rest_framework import routers
from rest_framework.schemas import get_schema_view
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView, TokenVerifyView, TokenBlacklistView

from . import views

schema_view = get_schema_view(
    title='My Hydroponic IoT',
    description='An API for Hydroponic IoT',
    version='1.0.0',
)

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)

urlpatterns = [
    path('api/', include(router.urls), name='api'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.LogoutView.as_view(), name='logout'),
    path('profile/', views.ProfileView.as_view(), name='profile'),
    path('odac/plant-disease/', views.ODACPlantDiseaseView.as_view(), name='odac'),
    path('odac/plant-disease/images/<path:path>', views.ODACPlantDiseaseImagesView.as_view(), name='odac'),
    path('forecasts/', views.ForecastImagesView.as_view(), name='forecast_images'),
    path('index/', views.index, name='index'),
    path('records/', views.RecordView.as_view(), name='records'),
    path('commands/', views.CommandView.as_view(), name='commands'),
    path('openapi/schema/', schema_view),
]
