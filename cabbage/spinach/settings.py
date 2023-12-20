#!/usr/bin/env python3
import os
import dotenv

dotenv.load_dotenv('.env')

WORKDIR = os.path.dirname(os.path.abspath(__file__))
ODAC_PLANT_DISEASE_SECRET = os.environ.get('ODAC_PLANT_DISEASE_SECRET', '')
IOT_HYDROPONIC_PROJECT_SECRET = os.environ.get('IOT_HYDROPONIC_PROJECT_SECRET', '')
