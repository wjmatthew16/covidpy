import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from collections import defaultdict

# Infection sensitivity by age group
age_multiplier = np.array([0.0, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8])

# Age to group
def classify_age_to_group_np(ages):
    ages = np.array(ages)
    age_class = np.zeros(len(ages), dtype=int)
    age_class[ages <= 29] = 1
    age_class[(30 <= ages) & (ages <= 39)] = 2
    age_class[(40 <= ages) & (ages <= 49)] = 3
    age_class[(50 <= ages) & (ages <= 59)] = 4
    age_class[(60 <= ages) & (ages <= 69)] = 5
    age_class[ages >= 70] = 6
    return age_class

# Mask transmission probability
def get_mask_prob(mi, mj):
    table = {
        ('x', 'x'): 0.9, ('x', 'c'): 0.7, ('x', 's'): 0.5, ('x', 'k'): 0.2,
        ('c', 'x'): 0.4, ('s', 'x'): 0.2, ('k', 'x'): 0.1,
        ('k', 'k'): 0.01, ('s', 'k'): 0.03, ('c', 'c'): 0.5,
    }
    return table.get((mi, mj), table.get((mj, mi), 0.5))

def compute_prob_np(mask_prob, dist, time, age_group, alpha=1.0, beta=0.2):
    base = mask_prob * np.exp(-alpha * dist) * (1 - np.exp(-beta * time))
    return np.minimum(base * age_multiplier[age_group], 1.0)

def simulate_seir_np(n, age_class, contacts_by_day, initially_infected,
                     days=10, exposure_period=2, infectious_period=5):
    status = np.full(n + 1, 'S', dtype='<U1')
    exposed_time = np.full(n + 1, -1)
    infected_time = np.full(n + 1, -1)

    status[initially_infected] = 'I'
    infected_time[initially_infected] = 0

    history = []
    for day in range(days):
        counts = [(status == s).sum() for s in ['S', 'E', 'I', 'R']]
        history.append(counts)

        exposed_to_infect = (status == 'E') & (day - exposed_time >= exposure_period)
        status[exposed_to_infect] = 'I'
        infected_time[exposed_to_infect] = day

        infect_to_recover = (status == 'I') & (day - infected_time >= infectious_period)
        status[infect_to_recover] = 'R'

        for i, j, mi, mj, dist, t in contacts_by_day[day]:
            for a, b, ma, mb in [(i, j, mi, mj), (j, i, mj, mi)]:
                if status[a] == 'I' and status[b] == 'S':
                    prob = compute_prob_np(get_mask_prob(ma, mb), dist, t, age_class[b])
                    if random.random() < prob:
                        status[b] = 'E'
                        exposed_time[b] = day

    history.append([(status == s).sum() for s in ['S', 'E', 'I', 'R']])
    return np.array(history)

# === input.txt based execution ===
with open('input.txt') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

n = int(lines[0])
k = int(lines[1])
initial = list(map(int, lines[2].split()))
ages = list(map(int, lines[3].split()))
age_class = np.insert(classify_age_to_group_np(ages), 0, 0)
days = int(lines[4])
m = int(lines[5])
contacts_by_day = defaultdict(list)

for line in lines[6:6 + m]:
    d, i, j, mi, mj, dist, t = line.split()
    contacts_by_day[int(d)].append((int(i), int(j), mi, mj, float(dist), float(t)))

for d in range(days):
    if d not in contacts_by_day:
        contacts_by_day[d] = []

trials = 1000
results = np.zeros((trials, days + 1, 4))
for trial in range(trials):
    results[trial] = simulate_seir_np(n, age_class, contacts_by_day, initial, days)

avg_result = results.mean(axis=0)
std_result = results.std(axis=0)

# Save to output.txt
df = pd.DataFrame(avg_result, columns=['S', 'E', 'I', 'R'])
df['Day'] = range(days + 1)
df = df[['Day', 'S', 'E', 'I', 'R']]
df.to_csv('output.txt', index=False, float_format='%.3f')

# Plot with standard deviation (error bands)
labels = ['S (Susceptible)', 'E (Exposed)', 'I (Infectious)', 'R (Recovered)']
colors = ['blue', 'orange', 'red', 'green']

for i in range(4):
    plt.plot(range(days + 1), avg_result[:, i], label=labels[i], color=colors[i])
    plt.fill_between(range(days + 1),
                     avg_result[:, i] - std_result[:, i],
                     avg_result[:, i] + std_result[:, i],
                     alpha=0.2, color=colors[i])

plt.xlabel('Day')
plt.ylabel('Number of People')
plt.title('SEIR Simulation (NumPy + input.txt with Std Deviation)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output_graph.png")
plt.close()
