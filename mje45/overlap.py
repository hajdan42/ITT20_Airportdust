import pandas as pd
import typing


DATA = pd.read_csv('../data/flight_log.csv')


def number_in_out(engine_number: int, airport_code: str) -> typing.Tuple[int, int]:
    engine_data = DATA[DATA["Engine No"] == engine_number]
    num_from = len(engine_data[engine_data["CITYPRFR"] == airport_code])
    num_to = len(engine_data[engine_data["CITYPRTO"] == airport_code])
    return num_from, num_to


def airport_list(engine_number: int, lower_limit = 100) -> typing.Tuple[list, list]:
    engine_data = DATA[DATA["Engine No"] == engine_number]
    unique_from = set(pd.unique(engine_data["CITYPRFR"]))
    unique_to = set(pd.unique(engine_data["CITYPRTO"]))
    over_lim_from = []
    over_lim_to = []
    for airport_code in list(unique_from.union(unique_to)):
        _from, _to = number_in_out(engine_number, airport_code)
        if _from >= lower_limit:
            over_lim_from.append(airport_code)
        if _to >= lower_limit:
            over_lim_to.append(airport_code)

    return over_lim_from, over_lim_to
