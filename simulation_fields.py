import pandas as pd

df = pd.read_csv('data/cleaned_df.csv')
df['created'] = pd.to_datetime(df['created'])
df['ended'] = pd.to_datetime(df['ended'])

# Calculate charge duration in minutes
df['charge_duration'] = (df['ended'] - df['created']).dt.total_seconds() / 60

# Estimate arrival rate by counting unique sessions per hour (or day)
df['date_hour'] = df['created'].dt.floor('H')  # Group by hour
arrival_rate_per_hour = df.groupby('date_hour').size().mean()  # Average sessions per hour

# Charge duration statistics
average_charge_duration = df['charge_duration'].mean()
std_dev_charge_duration = df['charge_duration'].std()

print(f"Average number of sessions per hour: {arrival_rate_per_hour}")
print(f"Average charge duration: {average_charge_duration} minutes")
print(f"Standard deviation of charge duration: {std_dev_charge_duration} minutes")
