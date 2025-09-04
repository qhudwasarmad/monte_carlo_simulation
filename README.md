Train Delay & Revenue Loss Simulation

This project provides a Monte Carlo simulation model for analysing train delays, cancellations, and revenue losses across UK stations. The model is calibrated using historical benchmarks (average delays and cancellation rates from ORR approximations, 2022–23) and simulates the impact of different disruption factors on both operational performance and compensation claims.

Key Features

Calibrated delay model: Uses a mixed approach combining a scaled Beta distribution for non-cancellation delays and fixed 120-minute delays for cancellations.

Cause-based revenue modelling: Samples revenue loss percentages from Gamma distributions parameterised by disruption causes (e.g., weather, congestion, staffing shortages).

Compensation tracking: Automatically counts events that trigger compensation under standard Delay Repay thresholds (15, 30, 60 minutes, and cancellations).

Validation against history: Simulated results (average delay and cancellation rate) are compared to historical station data, with error metrics included.

Customisable: Station data, disruption factors, and trial counts can be easily updated.

Outputs

For each station (Leeds, Manchester, Liverpool by default), the script prints a summary table with:

Simulated vs. historical average delay (minutes)

Error % in average delay

Simulated vs. historical cancellation rates

Error in cancellation rates (percentage points)

Average simulated revenue loss (%)

Raw arrays of simulated delays and losses are also returned for deeper analysis.
