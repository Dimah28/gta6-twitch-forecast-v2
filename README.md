# gta6-twitch-forecast-v2

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Dimah28/gta6-twitch-forecast-v2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Dimah28/gta6-twitch-forecast-v2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2–24 UTC, 19 Nov 2026) as five log-space quantiles per hour, anchored on the hour-1 print. It fits two estimators at run time from the case's own comparables — a quadratic OLS of log viewers on log hour-1 viewers and a 10-nearest-neighbours median in launch size — blended 0.7/0.3, trained only on tier-"comparable" big launches (hour-1 ≥ 10,000 viewers), with a shape-steepening correction for the diurnal rise, a small level shift, and a calibrated log-spread. Practice pinball loss 0.133 vs the 0.387 house persistence baseline (v1 scored 0.163 on the same four practice launches).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Stdlib only, no network, deterministic; reads every `<id>.in` in the input directory and writes `<id>.out` (`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`) to the output directory. See `main.py` for the full method and validation notes.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
