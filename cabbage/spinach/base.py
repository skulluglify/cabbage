from pydantic import BaseModel


class Record(BaseModel):
    ph: float
    ph_fuzzy: float
    ec: float
    air_temp: float
    humidity: float
    water_flow: bool
    lighting: bool
    full_water_tank: bool
    acid_actuator: bool
    alkaline_actuator: bool
    nutrient_actuator: bool
    fans_rpm: int


class Command(BaseModel):
    lighting: bool
    acid_actuator: bool
    alkaline_actuator: bool
    nutrient_actuator: bool
    fans_rpm: int
    accepted: bool
