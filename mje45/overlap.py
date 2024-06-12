import pandas as pd
import numpy as np
import scipy.sparse as ss
import typing


DATA = pd.read_csv('../data/flight_log.csv', keep_default_na=False)
GROUND_TRUTH = pd.read_csv('../data/ground_truth.csv', keep_default_na=False)
Y_DATA = pd.read_csv('../data/engines.csv', keep_default_na=False)


def number_in_out(engine_number: int, airport_code: str) -> typing.Tuple[int, int]:
    """
    For a given engine number and airport code, the output is a tuple containing the numer of flights from said airport
    and the number of flights to said airport.
    :param engine_number: Required engine number.
    :param airport_code: Required airport code.
    :return:
    """
    engine_data = DATA[DATA['Engine No'] == engine_number]
    num_from = len(engine_data[engine_data['CITYPRFR'] == airport_code])
    num_to = len(engine_data[engine_data['CITYPRTO'] == airport_code])
    return num_from, num_to


def top_airport_dict(engine_number: int, lower_limit: int = 100) -> typing.Tuple[dict, dict]:
    """
    Given a specific engine number, the output is a tuple containing two lists containing the airport code with over
    lower_limit many flights from and to respectively.
    :param engine_number: The engine number required
    :param lower_limit: The amount of flights from/to an airport to be counted.
    :return:
    """
    engine_data = DATA[DATA['Engine No'] == engine_number]
    unique_from = set(pd.unique(engine_data['CITYPRFR']))
    unique_to = set(pd.unique(engine_data['CITYPRTO']))
    over_lim_from = {}
    over_lim_to = {}
    for airport_code in list(unique_from.union(unique_to)):
        _from, _to = number_in_out(engine_number, airport_code)
        if _from >= lower_limit:
            over_lim_from[airport_code] = _from
        if _to >= lower_limit:
            over_lim_to[airport_code] = _to

    return over_lim_from, over_lim_to


def extract_all_airports():
    airports = list(GROUND_TRUTH['Airports'])
    return airports


AIRPORTS = extract_all_airports()


def generate_n_i(engine_number: int, lower_limit: int = 100):
    top_airports_from, _ = top_airport_dict(engine_number, lower_limit)
    # n_i = [(top_airports_from[airport_code] if airport_code in
    # top_airports_from.keys() else 0) for airport_code in AIRPORTS]
    ind_0, ind_1, data = [], [], []
    total_flights = sum(list(top_airports_from.values()))
    for i, airport_code in enumerate(AIRPORTS):
        if airport_code in top_airports_from:
            ind_0 += [5 * (engine_number - 1),
                      5 * (engine_number - 1) + 1,
                      5 * (engine_number - 1) + 2,
                      5 * (engine_number - 1) + 3,
                      5 * (engine_number - 1) + 4]
            ind_1 += [5 * i,
                      5 * i + 1,
                      5 * i + 2,
                      5 * i + 3,
                      5 * i + 4]
            data += 5 * [float(top_airports_from[airport_code]) / float(total_flights)]

    return ind_0, ind_1, data


def generate_a(lower_limit: int = 100):
    ind_0, ind_1, data = [], [], []
    no_engines = len(pd.unique(DATA['Engine No']))
    for engine_no in pd.unique(DATA['Engine No']):
        id0, id1, _data = generate_n_i(engine_no, lower_limit)
        ind_0 += id0
        ind_1 += id1
        data += _data

    return ss.csr_matrix((data, (ind_0, ind_1)), shape=(5 * no_engines, 5 * len(AIRPORTS)))


def generate_y() -> np.ndarray:
    y_pure_data = Y_DATA.drop('Engine No', axis='columns')
    y_pure_data = y_pure_data.to_numpy()
    y = np.atleast_2d(y_pure_data.flatten("C")).T
    return y


A = generate_a(lower_limit=50)
b = generate_y()

x_pure_data = GROUND_TRUTH.drop('Airports', axis='columns')
x_pure_data = x_pure_data.to_numpy()

x = ss.linalg.lsqr(A, b)[0]
x = np.atleast_2d(x).reshape((-1, 5), order="C")

print(np.linalg.norm(x - x_pure_data))
