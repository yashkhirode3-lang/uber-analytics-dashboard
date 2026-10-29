"""Generate a synthetic dataset that mirrors the schema of the Kaggle
"Uber Ride Analytics Dashboard" file (ncr_ride_bookings.csv).

Use this ONLY to try the app offline. For the real project, download the CSV
from Kaggle and place it at  data/ncr_ride_bookings.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 40_000

VEHICLES = ["Auto", "Go Mini", "Go Sedan", "Bike", "Premier Sedan", "eBike", "Uber XL"]
RATE = {"Auto": 14, "Go Mini": 16, "Go Sedan": 20, "Bike": 9, "Premier Sedan": 26, "eBike": 8, "Uber XL": 32}
LOCS = ["Connaught Place", "Gurgaon Sector 29", "Noida Sector 62", "Saket", "Dwarka", "Karol Bagh",
        "Lajpat Nagar", "Hauz Khas", "Rohini", "Vasant Kunj", "Nehru Place", "Janakpuri",
        "Chandni Chowk", "Airport T3", "Cyber Hub", "Pitampura"]
PAY = ["UPI", "Cash", "Uber Wallet", "Credit Card", "Debit Card"]
CUST_REASONS = ["Change of plans", "Driver is not moving towards pickup", "Wrong Address",
                "Driver asked to cancel", "AC is not working"]
DRV_REASONS = ["Customer related issue", "Personal & Car related issue", "More than permitted people in there",
               "Customer Demand"]

dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
# hour-of-day demand with morning & evening peaks
hour_p = np.array([1, .5, .4, .3, .5, 1.5, 3, 5, 6, 4, 3, 3, 3.5, 3.5, 3, 3.5, 4.5, 6, 6.5, 5.5, 4, 3, 2, 1.5])
hour_p = hour_p / hour_p.sum()
hours = rng.choice(24, N, p=hour_p)
mins = rng.integers(0, 60, N)
secs = rng.integers(0, 60, N)
time = [f"{h:02d}:{m:02d}:{s:02d}" for h, m, s in zip(hours, mins, secs)]

veh = rng.choice(VEHICLES, N, p=[.18, .17, .16, .14, .11, .12, .12])
pick = rng.choice(LOCS, N)
drop = np.array([rng.choice([l for l in LOCS if l != p]) for p in pick])
dist = np.clip(rng.gamma(2.4, 6.5, N), 1, 50).round(2)
peak = np.isin(hours, [8, 9, 17, 18, 19]).astype(int)
vtat = np.clip(rng.normal(8.5 + 3 * peak, 3), 2, 25).round(1)   # vehicle time-to-arrive
ctat = np.clip(rng.normal(28 + dist * .8, 6), 8, 60).round(1)    # customer time-at-trip

value = np.array([RATE[v] for v in veh]) * dist * (1 + .25 * peak) + 40 + rng.normal(0, 25, N)
value = np.clip(value, 30, None).round(0)

# status model: longer VTAT + peak + long distance -> more cancellations
p_cancel_c = 1 / (1 + np.exp(-(-2.6 + .09 * (vtat - 8) + .3 * peak + .015 * dist)))
p_cancel_d = 1 / (1 + np.exp(-(-3.0 + .25 * peak + .02 * (veh == "Uber XL"))))
u = rng.random(N)
status = np.where(u < p_cancel_c, "Cancelled by Customer",
          np.where(u < p_cancel_c + p_cancel_d, "Cancelled by Driver",
          np.where(u < p_cancel_c + p_cancel_d + .05, "No Driver Found",
          np.where(u < p_cancel_c + p_cancel_d + .05 + .04, "Incomplete", "Completed"))))

def sw(mask, choices, p=None):
    """string column that is NaN where mask is False"""
    vals = rng.choice(choices, N, p=p).astype(object)
    vals[~mask] = np.nan
    return vals


df = pd.DataFrame({
    "Date": dates.strftime("%Y-%m-%d"),
    "Time": time,
    "Booking ID": [f'"CNR{1000000 + i}"' for i in range(N)],
    "Booking Status": status,
    "Customer ID": [f'"CID{rng.integers(1, 5_000_000)}"' for _ in range(N)],
    "Vehicle Type": veh,
    "Pickup Location": pick,
    "Drop Location": drop,
    "Avg VTAT": vtat,
    "Avg CTAT": ctat,
    "Cancelled Rides by Customer": np.where(status == "Cancelled by Customer", 1, np.nan),
    "Reason for cancelling by Customer": sw(status == "Cancelled by Customer", CUST_REASONS),
    "Cancelled Rides by Driver": np.where(status == "Cancelled by Driver", 1, np.nan),
    "Driver Cancellation Reason": sw(status == "Cancelled by Driver", DRV_REASONS),
    "Incomplete Rides": np.where(status == "Incomplete", 1, np.nan),
    "Incomplete Rides Reason": sw(status == "Incomplete", ["Vehicle Breakdown", "Other Issue", "Customer Demand"]),
    "Booking Value": np.where(status == "Completed", value, np.nan),
    "Ride Distance": np.where(status == "Completed", dist, np.nan),
    "Driver Ratings": np.where(status == "Completed", np.clip(rng.normal(4.2, .5, N), 3, 5).round(1), np.nan),
    "Customer Rating": np.where(status == "Completed", np.clip(rng.normal(4.4, .45, N), 3, 5).round(1), np.nan),
    "Payment Method": sw(status == "Completed", PAY, [.45, .22, .12, .11, .10]),
})
out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)
df.to_csv(out / "sample_ncr_ride_bookings.csv", index=False)
print("wrote", out / "sample_ncr_ride_bookings.csv", df.shape)
