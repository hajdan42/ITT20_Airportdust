import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.sparse as ss
import scipy.optimize as so
import typing
import matplotlib.pyplot
import warnings

DATA = pd.read_csv('../data/flight_log.csv', keep_default_na=False)
GROUND_TRUTH = pd.read_csv('../data/ground_truth.csv', keep_default_na=False)
Y_DATA = pd.read_csv('../data/engines.csv', keep_default_na=False)
LOWER_LIMIT = 350
NO_ENGINES = len(pd.unique(DATA['Engine No']))


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


def valid_airports(lower_limit: int = 100) -> typing.List[str]:
    considered_airports = set()
    for i in range(NO_ENGINES):
        top_from, _ = top_airport_dict(i, lower_limit)
        if (new_airports := set(top_from.keys())) is not None:
            considered_airports = considered_airports.union(new_airports)

    valid_airport_list = list(considered_airports)
    valid_airport_list.sort()

    return valid_airport_list


AIRPORTS = extract_all_airports()
VALID_AIRPORTS = valid_airports(lower_limit=LOWER_LIMIT)


def generate_n_i(engine_number: int, lower_limit: int = 100) -> typing.Optional[list]:
    top_airports_from, _ = top_airport_dict(engine_number, lower_limit)
    n_i = [(top_airports_from[airport_code] if (airport_code in top_airports_from) else 0) for airport_code in VALID_AIRPORTS]

    if s_i := sum(n_i):
        n_i = list(np.array(n_i) / s_i)
    else:
        Y_DATA.drop(axis=0, labels=engine_number)
    return n_i


def generate_a(lower_limit: int = 100):
    a = np.kron(generate_n_i(1, lower_limit), np.eye(5, dtype=int))  # could be shite
    for i in range(NO_ENGINES - 1):
        if (n_i := generate_n_i(i+2, lower_limit)) is not None:
            a = np.concatenate((a, np.kron(n_i, np.eye(5, dtype=int))), axis=0)

    return a


def generate_y() -> np.ndarray:
    y_pure_data = Y_DATA.drop('Engine No', axis='columns')
    y_pure_data = y_pure_data.to_numpy()
    y = np.atleast_2d(y_pure_data.flatten("C")).T
    return y


print(DATA['Engine No'].value_counts())

warnings.filterwarnings('ignore', '', FutureWarning)

A = generate_a(lower_limit=LOWER_LIMIT)
b = generate_y()

print(A.shape, A.sum(axis=1))

plt.imshow(A)
plt.show()

mask = GROUND_TRUTH['Airports'].isin(VALID_AIRPORTS)
x_pure_data = GROUND_TRUTH[mask]
x_pure_data = x_pure_data.drop('Airports', axis=1)
x_pure_data = x_pure_data.to_numpy()

# Section: Pseudo-inverse approach

x_out = np.linalg.pinv(A) @ b
x = np.atleast_2d(x_out).reshape((-1, 5), order="C")

print(x)

plt.imshow(np.abs(x - x_pure_data))
plt.colorbar()
plt.show()

print(np.linalg.norm(x - x_pure_data))


# Section: Constrained optimisation approach

NO_VALID_AIRPORTS = len(VALID_AIRPORTS)

eq_cons = [{'type': 'eq', 'fun': lambda _x: _x[5 * j] + _x[5 * j + 1] + _x[5 * j + 2] +
                    _x[5 * j + 3] + _x[5 * j + 4] - 100.0} for j in range(NO_VALID_AIRPORTS)]
ineq_cons = [{'type': 'ineq', 'fun': lambda _x: _x[i]} for i in range(5 * NO_VALID_AIRPORTS)]
cons = tuple(eq_cons + ineq_cons)

# x_0 = 20 * np.ones((5 * NO_VALID_AIRPORTS, 1), dtype=float)
x_0 = x_out

statement = lambda _x: np.linalg.norm(A @ _x - b) ** 2
statement_jac = lambda _x: 2.0 * A.T @ A @ _x - 2.0 * A.T @ b

bounds = tuple([(0, 100) for _ in range(5 * NO_VALID_AIRPORTS)])

# res = so.minimize(statement, x_0, method='SLSQP', jac=statement_jac,
#                   constraints=[ineq_cons, eq_cons],
#                   options={'ftol': 1e-9, 'disp': True, 'verbose': 1})
res = so.minimize(statement, x_0, bounds=bounds, constraints=cons)

x = res.x
x = np.atleast_2d(x).reshape((-1, 5), order="C")

plt.imshow(np.abs(x - x_pure_data))
plt.colorbar()
plt.show()

print(np.linalg.norm(x - x_pure_data))
