import numpy as np
import pandas as pd
from scipy.stats import beta, gamma

"""
Improved simulation model for train delays and revenue loss.

This script calibrates delay distributions to match historical average
delays and cancellation rates for each station by constructing a mixed
model.  For each station we compute the mean delay of non-cancellation
events (mu) based on the historical average delay (hist_avg) and the
cancellation rate (hist_cxl):

    mu = (hist_avg - hist_cxl * 120.0) / (1.0 - hist_cxl)

We then simulate non-cancellation delays using a Beta(2,6) distribution
scaled to the interval [0, 4*mu].  The expected value of a Beta(2,6)
distribution is 2/(2+6) = 0.25, so the mean of the scaled distribution
is 0.25 * 4*mu = mu.  This ensures that the mean delay of all events
(cancellations and non-cancellations) matches the historical average
when the cancellation rate is applied.

Cancellation events are introduced with probability equal to the
historical cancellation rate.  When a cancellation occurs, the delay is
set to 120 minutes.  The revenue loss for cancellations is drawn from
a Gamma distribution with shape parameter equal to twice the sampled
gamma_alpha value; non-cancellations use a Gamma distribution with
shape equal to the sampled gamma_alpha.  In both cases the scale
parameter is set to 1.0, so the mean percentage loss is proportional
to the shape parameter.

The simulation also computes compensation claim counts for the standard
Delay Repay thresholds (15, 30, 60 minutes) and cancellations.
"""

# Historical benchmarks (ORR approximations, 2022–23)
historical_data = {
    "Leeds": {"avg_delay": 8.5, "cancellation_rate": 0.025},
    "Manchester": {"avg_delay": 10.2, "cancellation_rate": 0.031},
    "Liverpool": {"avg_delay": 7.4, "cancellation_rate": 0.018},
}

COMPENSATION_THRESHOLDS = {
    "Delay Repay 15": 15,
    "Delay Repay 30": 30,
    "Delay Repay 60": 60,
    "Cancellation": 120
}

# Factor definitions for gamma distribution shapes; used to sample loss
# percentages.  Probabilities reflect relative incidence of each cause.
stations_factors = {
    "Leeds": {
        "Weather": {"prob": 0.35, "gamma_alpha": 5},
        "Signal Failure": {"prob": 0.25, "gamma_alpha": 4},
        "Technical Issues": {"prob": 0.20, "gamma_alpha": 2},
        "Staffing Shortages": {"prob": 0.15, "gamma_alpha": 3},
        "Congestion": {"prob": 0.05, "gamma_alpha": 5},
    },
    "Manchester": {
        "Congestion": {"prob": 0.40, "gamma_alpha": 5},
        "Technical Issues": {"prob": 0.30, "gamma_alpha": 2},
        "Signal Failure": {"prob": 0.15, "gamma_alpha": 4},
        "Weather": {"prob": 0.10, "gamma_alpha": 5},
        "Staffing Shortages": {"prob": 0.05, "gamma_alpha": 3},
    },
    "Liverpool": {
        "Staffing Shortages": {"prob": 0.30, "gamma_alpha": 3},
        "Signal Failure": {"prob": 0.25, "gamma_alpha": 4},
        "Technical Issues": {"prob": 0.20, "gamma_alpha": 2},
        "Congestion": {"prob": 0.15, "gamma_alpha": 5},
        "Weather": {"prob": 0.10, "gamma_alpha": 5},
    }
}

def simulate_station(station_name: str, n_trials: int = 10000, random_state: int = None):
    """
    Simulate delays and revenue losses for a single station.

    Parameters
    ----------
    station_name : str
        Name of the station (must be a key in historical_data and stations_factors).
    n_trials : int
        Number of simulated events (default 10000).
    random_state : int or None
        Seed for the random number generator.  If None, seed is not set.

    Returns
    -------
    dict
        Dictionary containing summary statistics and raw simulation results.
    """
    if random_state is not None:
        np.random.seed(random_state)

    hist_avg = historical_data[station_name]["avg_delay"]
    hist_cxl = historical_data[station_name]["cancellation_rate"]
    # Mean delay for non-cancellation events
    mu = (hist_avg - hist_cxl * 120.0) / (1.0 - hist_cxl)
    # Scale parameter for Beta distribution (non-cancellations)
    B = 4.0 * mu
    # Beta shape parameters (alpha, beta) chosen to produce mean 0.25
    beta_alpha = 2.0
    beta_beta = 6.0
    # Prepare factor distribution for gamma_alpha sampling
    factor_names = list(stations_factors[station_name].keys())
    factor_probs = np.array([stations_factors[station_name][fn]["prob"] for fn in factor_names])
    factor_probs /= factor_probs.sum()
    factor_gamma = np.array([stations_factors[station_name][fn]["gamma_alpha"] for fn in factor_names])
    delays = []
    losses = []
    cancellations = 0
    compensation_counts = {k: 0 for k in COMPENSATION_THRESHOLDS.keys()}

    for _ in range(n_trials):
        # Determine if this event is a cancellation
        if np.random.rand() < hist_cxl:
            # Cancellation event
            delay = 120.0
            cancellations += 1
            # Sample gamma_alpha according to factor probabilities
            gamma_alpha = np.random.choice(factor_gamma, p=factor_probs)
            loss = gamma.rvs(gamma_alpha * 2.0, scale=1.0)
        else:
            # Non-cancellation event: sample delay from Beta distribution
            beta_sample = beta.rvs(beta_alpha, beta_beta)
            delay = beta_sample * B
            # Sample gamma_alpha according to factor probabilities
            gamma_alpha = np.random.choice(factor_gamma, p=factor_probs)
            loss = gamma.rvs(gamma_alpha, scale=1.0)
        delays.append(delay)
        losses.append(loss)
        # Count compensation claims for thresholds
        for comp_name, threshold in sorted(COMPENSATION_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if delay >= threshold:
                compensation_counts[comp_name] += 1
                break
    # Convert results to arrays for efficiency
    delays = np.array(delays)
    losses = np.array(losses)
    # Compute average delay and cancellation rate
    sim_avg_delay = delays.mean()
    sim_cxl_rate = cancellations / n_trials
    # Compute average loss
    sim_avg_loss = losses.mean()
    # Prepare summary dictionary
    return {
        "avg_delay": sim_avg_delay,
        "avg_loss": sim_avg_loss,
        "max_delay": delays.max(),
        "cancellation_rate": sim_cxl_rate,
        "compensation_counts": compensation_counts,
        "delays": delays,
        "losses": losses
    }

def main():
    np.random.seed(42)
    n_trials = 10000
    results = {}
    summary_data = []
    for station in historical_data.keys():
        res = simulate_station(station, n_trials=n_trials)
        results[station] = res
        hist_avg = historical_data[station]["avg_delay"]
        hist_cxl = historical_data[station]["cancellation_rate"]
        summary_data.append({
            "Station": station,
            "Sim Avg Delay (mins)": f"{res['avg_delay']:.2f}",
            "Hist Avg Delay (mins)": f"{hist_avg:.2f}",
            "Delay Error (%)": f"{100.0 * (res['avg_delay'] - hist_avg) / hist_avg:.2f}",
            "Sim Cancellation Rate (%)": f"{res['cancellation_rate'] * 100.0:.2f}",
            "Hist Cancellation Rate (%)": f"{hist_cxl * 100.0:.2f}",
            "Cancellation Error (pp)": f"{100.0 * (res['cancellation_rate'] - hist_cxl):.2f}",
            "Avg Revenue Loss (%)": f"{res['avg_loss']:.2f}"
        })
    df = pd.DataFrame(summary_data)
    print("\nImproved Simulation Summary:\n")
    # Use DataFrame.to_string instead of to_markdown to avoid requiring tabulate
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()