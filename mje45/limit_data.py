import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as so
import typing
import warnings
import statsmodels.api as sm

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
    """
    Extracts the full list of airports availible
    :return:
    """
    airports = list(GROUND_TRUTH['Airports'])
    return airports


def valid_airports(lower_limit: int = 100) -> typing.List[str]:
    """
    From a given lower limit, we take all the airports from all engines which have at least ```lower_limit``` number of
    flights from.
    :param lower_limit: (int) represents the least number of flights one needs before deeming an airport significant.
    :return: List of airports of form ```typing.List[str]```
    """
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
    """
    For a given engine and lower limit, this calculates the distribution of visits to each of the valid airports.
    :param engine_number: (int) of the datu, in question
    :param lower_limit: (int) represents the least number of flights one needs before deeming an airport significant.
    :return:
    """
    top_airports_from, _ = top_airport_dict(engine_number, lower_limit)
    n_i = [
        (top_airports_from[airport_code] if (airport_code in top_airports_from) else 0)
        for airport_code in VALID_AIRPORTS
    ]

    if s_i := sum(n_i):
        n_i = list(np.array(n_i) / s_i)
    else:
        Y_DATA.drop(axis=0, labels=engine_number)
    return n_i


def generate_a(lower_limit: int = 100):
    """
    Generates the linear system's matrix, A.
    :param lower_limit: (int) represents the least number of flights one needs before deeming an airport significant.
    :return:
    """
    a = np.kron(generate_n_i(1, lower_limit), np.eye(5, dtype=int))  # could be shite
    for i in range(NO_ENGINES - 1):
        if (n_i := generate_n_i(i+2, lower_limit)) is not None:
            a = np.concatenate((a, np.kron(n_i, np.eye(5, dtype=int))), axis=0)

    return a


def generate_y() -> np.ndarray:
    """
    Generates the linear system's b vector (Ax=b) out of all the given engine data.
    :return:
    """
    y_pure_data = Y_DATA.drop('Engine No', axis='columns')
    y_pure_data = y_pure_data.to_numpy()
    y = np.atleast_2d(y_pure_data.flatten("C")).T
    return y


print(DATA['Engine No'].value_counts())
warnings.filterwarnings('ignore', '', FutureWarning)

A = generate_a(lower_limit=LOWER_LIMIT)
b = generate_y()

print("Shape of A:")
print(A.shape, A.sum(axis=1))

plt.title("Structure of A (Elements)")
plt.imshow(A)
plt.colorbar()
plt.show()

mask = GROUND_TRUTH['Airports'].isin(VALID_AIRPORTS)
x_pure_data = GROUND_TRUTH[mask]
x_pure_data = x_pure_data.drop('Airports', axis=1)
x_pure_data = x_pure_data.to_numpy()

# Section: Pseudo-inverse approach

# This generates a solution that minimises \|Ax-b\|_2 but doesn't necessarily meet the constraints that we require. This
# is however a reasonable guess at the optimal solution given the constraints and we can use this as an initial
# condition in a constrained optimisation solver.

x_out = np.linalg.pinv(A) @ b
# x = np.atleast_2d(x_out).reshape((-1, 5), order="C")

# plt.title("Pseudo-inverse Solution: \nabsolute errors [%]")
# plt.xticks(range(5), ["C", "M", "A", "S", "O"])
# plt.yticks(range(len(VALID_AIRPORTS)), VALID_AIRPORTS)
# plt.imshow(np.abs(x - x_pure_data))
# plt.colorbar()
# plt.show()

# Section: Constrained optimisation approach

# This solution takes in the pseudo-inverse solution as an initial condition and contrains the optimisation of the same
# loss function.

NO_VALID_AIRPORTS = len(VALID_AIRPORTS)

eq_cons = [{'type': 'eq', 'fun': lambda _x: _x[5 * j] + _x[5 * j + 1] + _x[5 * j + 2] +
                    _x[5 * j + 3] + _x[5 * j + 4] - 100.0} for j in range(NO_VALID_AIRPORTS)]
ineq_cons = [{'type': 'ineq', 'fun': lambda _x: _x[i]} for i in range(5 * NO_VALID_AIRPORTS)]
cons = tuple(eq_cons + ineq_cons)

x_0 = x_out

statement = lambda _x: np.linalg.norm(A @ _x - b) ** 2
bounds = tuple([(0, 100) for _ in range(5 * NO_VALID_AIRPORTS)])
res = so.minimize(statement, x_0, bounds=bounds, constraints=cons)

x = res.x
x = np.atleast_2d(x).reshape((-1, 5), order="C")

abs_errors = np.abs(x - x_pure_data)
# abs_errors = (x - x_pure_data) / x


fig = plt.figure(layout="constrained", figsize=(12, 6))
subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[0.5, 1.])


axs0 = subfigs[1].subplots(2, 2)

subfigs[1].suptitle('EPDFs of the Errors')

c_cont = abs_errors[:, 0]
m_cont = abs_errors[:, 1]
a_cont = abs_errors[:, 2]
s_cont = abs_errors[:, 3]

kdec = sm.nonparametric.KDEUnivariate(c_cont)
# kdec.fit(clip=(0, np.inf))
kdec.fit()

axs0[0, 0].hist(c_cont, density=True, bins=15, alpha=0.5, color="#D58817")
axs0[0, 0].plot(kdec.support, kdec.density, color="#5AB4DC")
axs0[0, 0].annotate("C", xy=(0.9, 0.9), xycoords='axes fraction', style="normal", weight="bold", size=15)


kdem = sm.nonparametric.KDEUnivariate(m_cont)
# kdem.fit(clip=(0, np.inf))
kdem.fit()

axs0[0, 1].hist(m_cont, density=True, bins=15, alpha=0.5, color="#D58817")
axs0[0, 1].plot(kdem.support, kdem.density, color="#5AB4DC")
axs0[0, 1].annotate("M", xy=(0.9, 0.9), xycoords='axes fraction', style="normal", weight="bold", size=15)


kdea = sm.nonparametric.KDEUnivariate(a_cont)
# kdea.fit(clip=(0, np.inf))
kdea.fit()

axs0[1, 0].hist(a_cont, density=True, bins=15, alpha=0.5, color="#D58817")
axs0[1, 0].plot(kdea.support, kdea.density, color="#5AB4DC")
axs0[1, 0].annotate("A", xy=(0.9, 0.9), xycoords='axes fraction', style="normal", weight="bold", size=15)


kdes = sm.nonparametric.KDEUnivariate(s_cont)
# kdes.fit(clip=(0, np.inf))
kdes.fit()

axs0[1, 1].hist(s_cont, density=True, bins=15, alpha=0.5, color="#D58817")
axs0[1, 1].plot(kdes.support, kdes.density, color="#5AB4DC")
axs0[1, 1].annotate("S", xy=(0.9, 0.9), xycoords='axes fraction', style="normal", weight="bold", size=15)

axs1 = subfigs[0].subplots(1, 1)

axs1.set_title("Contrained Optimisation Solution: \nabsolute errors [%]")
axs1.set_xticks(range(5), ["C", "M", "A", "S", "O"])
axs1.set_yticks(range(len(VALID_AIRPORTS)), VALID_AIRPORTS)
axs1.set_ylabel('Airports')
image = axs1.imshow(abs_errors)
subfigs[0].colorbar(image)

plt.show()
