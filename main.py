"""Forecast GTA VI launch-day Twitch viewers, hours 2..24, from the hour-1 print.

Method (per case, deterministic, Python 3.13 standard library only):
  * Keep only past launches whose unlock note parses to tier "comparable",
    whose hour-1 audience is at least 10,000, and whose 24 hourly prints are
    all present and positive. The target game's own history is dropped.
  * For each hour h = 2..24, fit a quadratic least-squares curve of
    log(viewers at h) against log(viewers at hour 1) over the kept launches,
    and take the median log-viewers at h of the 10 kept launches whose
    log hour-1 is closest to the target's. Blend the two in log space,
    0.7 * quadratic fit + 0.3 * nearest-neighbour median.
  * Steepen the daily shape: deviations of the blended log-ratios
    (log viewers_h - log viewers_1) from their h = 2..24 mean are scaled by
    1.2. Then shift every hour's log forecast up by 0.05.
  * Spread: the per-hour residual scatter of the quadratic fit pooled with
    the scatter of the 10 neighbours' log values, shrunk by 0.55, gives the
    log-normal spread for the five quantiles (z = -1.645, -0.674, 0, 0.674,
    1.645). Log forecasts are clamped to the training range widened by
    1 log unit, median ratios to [0.1, 100], and spreads to [0.05, 1.5].
  * With no usable comparable at all, fall back to persistence: every hour
    forecast at the hour-1 print, with a widening default spread.

Input: every *.in file in $MAKEFUN_INPUT_DIR (else /input), one JSON case
each (schema makefun/twitch-ccu-case/v1). Output: one <id>.out per case in
$MAKEFUN_OUTPUT_DIR (else /output): {"hours": [{"hour": h,
"q": [q05, q25, q50, q75, q95]}, ...]} for h = 2..24.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = list(range(2, 25))

MIN_HOUR1 = 10000.0
NEIGHBOURS = 10
OLS_WEIGHT = 0.7
STEEPEN = 1.2
LEVEL_SHIFT = 0.05
SPREAD_SHRINK = 0.55
RATIO_LOW = 0.1
RATIO_HIGH = 100.0
SPREAD_LOW = 0.05
SPREAD_HIGH = 1.5
LOG_WIDEN = 1.0
MAX_VIEWERS = 1e9


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp ('...Z') into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def stamp(moment):
    """Format an aware datetime back to the kit's '...Z' timestamp form."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tier_of(comp):
    """The tier word after 'Tier:' in a comparable's unlock note, or ''."""
    note = comp.get("unlock") or ""
    if "Tier:" not in note:
        return ""
    tail = note.split("Tier:", 1)[1]
    return tail.split(".")[0].strip().lower()


def day_curve(comp):
    """The comparable's 24 hourly viewer counts, hours 1..24, or None.

    Returns None when any of the 24 release-day hours is missing or not
    positive, so only complete curves survive.
    """
    by_time = {}
    for row in comp.get("hourly") or []:
        try:
            by_time[row["from"]] = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
    try:
        start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    curve = []
    for h in range(1, 25):
        value = by_time.get(stamp(start + timedelta(hours=h - 1)))
        if value is None or value <= 0:
            return None
        curve.append(value)
    return curve


def usable_comparables(case):
    """Filtered training launches: (log hour-1, [log viewers hour 1..24])."""
    target_name = (case.get("game") or {}).get("name") or ""
    kept = []
    for comp in case.get("comparables") or []:
        if tier_of(comp) != "comparable":
            continue
        name = comp.get("name") or ""
        if name == target_name or tier_of(comp) == "target-history":
            continue
        curve = day_curve(comp)
        if curve is None:
            continue
        if curve[0] < MIN_HOUR1:
            continue
        kept.append((math.log(curve[0]), [math.log(v) for v in curve]))
    kept.sort(key=lambda item: item[0])
    return kept


def solve_3x3(matrix, rhs):
    """Solve a 3x3 linear system by Gaussian elimination with pivoting."""
    rows = [list(matrix[i]) + [rhs[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, 3):
            factor = rows[r][col] / rows[col][col]
            for c in range(col, 4):
                rows[r][c] -= factor * rows[col][c]
    out = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        total = rows[i][3] - sum(rows[i][j] * out[j] for j in range(i + 1, 3))
        out[i] = total / rows[i][i]
    return out


def quad_fit(xs, ys):
    """Least-squares coefficients [a, b, c] of y ~ a + b*x + c*x^2.

    Returns None when the design is degenerate (too few distinct x).
    """
    n = len(xs)
    if n < 3 or len(set(xs)) < 3:
        return None
    sums = [0.0] * 5
    cross = [0.0] * 3
    for x, y in zip(xs, ys):
        powers = [1.0, x, x * x, x ** 3, x ** 4]
        for i in range(5):
            sums[i] += powers[i]
        for i in range(3):
            cross[i] += powers[i] * y
    matrix = [[sums[0], sums[1], sums[2]],
              [sums[1], sums[2], sums[3]],
              [sums[2], sums[3], sums[4]]]
    return solve_3x3(matrix, cross)


def median(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def round2(value):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = value * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100.0
    return int(rounded) if float(rounded).is_integer() else rounded


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def forecast_hour(hour, x0, log_p1, training):
    """Median log-viewers and log-spread for one hour from the training set."""
    xs = [item[0] for item in training]
    ys = [item[1][hour - 1] for item in training]
    coefs = quad_fit(xs, ys)
    if coefs is None:
        ols = median(ys)
        residuals = [y - ols for y in ys]
    else:
        ols = coefs[0] + coefs[1] * x0 + coefs[2] * x0 * x0
        residuals = [y - (coefs[0] + coefs[1] * x + coefs[2] * x * x)
                     for x, y in zip(xs, ys)]
    nearest = sorted(training, key=lambda item: abs(item[0] - x0))[:NEIGHBOURS]
    neighbour_logs = [item[1][hour - 1] for item in nearest]
    nn_med = median(neighbour_logs)
    blended = OLS_WEIGHT * ols + (1.0 - OLS_WEIGHT) * nn_med
    # Pooled scatter: OLS residuals together with the neighbours' spread
    # around their own median, shrunk towards the centre.
    ss_res = sum(r * r for r in residuals)
    ss_nn = sum((v - nn_med) ** 2 for v in neighbour_logs)
    dof = max(1, (len(xs) - 3) + (len(neighbour_logs) - 1))
    spread = SPREAD_SHRINK * math.sqrt((ss_res + ss_nn) / dof)
    spread = clamp(spread, SPREAD_LOW, SPREAD_HIGH)
    train_logs = ys
    size_low = min(train_logs) - LOG_WIDEN
    size_high = max(train_logs) + LOG_WIDEN
    return blended, spread, size_low, size_high


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("levels are not [0.05, 0.25, 0.5, 0.75, 0.95]")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)
    training = usable_comparables(case)

    if not training:
        rows = []
        for h in HOURS:
            spread = 0.15 * math.sqrt(h - 1)
            quantiles = [round2(math.exp(log_p1 + z * spread)) for z in Z_SCORES]
            quantiles = [clamp(q, 1e-9, MAX_VIEWERS) for q in quantiles]
            rows.append({"hour": h, "q": quantiles})
        return {"hours": rows}

    raw = {}
    bounds = {}
    for h in HOURS:
        blended, spread, low, high = forecast_hour(h, log_p1, log_p1, training)
        raw[h] = (blended, spread)
        bounds[h] = (low, high)

    ratios = [raw[h][0] - log_p1 for h in HOURS]
    mean_ratio = sum(ratios) / len(ratios)
    rows = []
    for h in HOURS:
        blended, spread = raw[h]
        steep = mean_ratio + STEEPEN * ((blended - log_p1) - mean_ratio)
        steep = clamp(steep, math.log(RATIO_LOW), math.log(RATIO_HIGH))
        centre = clamp(log_p1 + steep + LEVEL_SHIFT,
                       bounds[h][0], bounds[h][1])
        quantiles = []
        prev = 0.0
        for z in Z_SCORES:
            q = clamp(round2(math.exp(centre + z * spread)), 1e-9, MAX_VIEWERS)
            q = max(q, prev)
            quantiles.append(q)
            prev = q
        rows.append({"hour": h, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(json.dumps(result, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
