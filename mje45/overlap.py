import pandas as pd
import typing


DATA = pd.read_csv('../data/flight_log.csv')


def number_in_out(engine_number: int, airport_code: str) -> typing.Tuple[int, int]:
    """
    For a given engine number and airport code, the output is a tuple containing the numer of flights from said airport
    and the number of flights to said airport.
    :param engine_number: Required engine number.
    :param airport_code: Required airport code.
    :return:
    """
    engine_data = DATA[DATA["Engine No"] == engine_number]
    num_from = len(engine_data[engine_data["CITYPRFR"] == airport_code])
    num_to = len(engine_data[engine_data["CITYPRTO"] == airport_code])
    return num_from, num_to


def top_airport_list(engine_number: int, lower_limit=100) -> typing.Tuple[list, list]:
    """
    Given a specific engine number, the output is a tuple containing two lists containing the airport code with over
    lower_limit many flights from and to respectively.
    :param engine_number: The engine number required
    :param lower_limit: The amount of flights from/to an airport to be counted.
    :return:
    """
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

    over_lim_from.sort()
    over_lim_to.sort()

    return over_lim_from, over_lim_to


print(top_airport_list(1))
