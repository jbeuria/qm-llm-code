import requests
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import sem
from itertools import combinations
from tqdm import tqdm
import seaborn as sns


OLLAMA_URL = "http://localhost:11434/api/generate"

# ------------------------------------------------
# MODELS
# ------------------------------------------------

MODELS = [
    "llama3",
    "mistral",
    "qwen2",
    "gemma"
]

TRIALS_PER_TASK = 1000
MAX_WORKERS = 20


# ------------------------------------------------
# NUMBER EXTRACTION
# ------------------------------------------------

def extract_number(text):

    nums = re.findall(r"\d*\.?\d+", text)

    if len(nums)==0:
        return None

    x = float(nums[0])

    if x > 1:
        x = x/100

    return max(0,min(1,x))


# ------------------------------------------------
# MODEL QUERY
# ------------------------------------------------

def query_model(model,prompt):

    instruction = """
Respond ONLY with a number between 0 and 1.
Do not explain.
"""

    payload = {
        "model":model,
        "prompt":instruction + "\n\n" + prompt,
        "stream":False,
        "options":{
            "temperature":0.7,
            "top_p":0.9,
            "num_predict":10
        }
    }

    try:

        r = requests.post(OLLAMA_URL,json=payload,timeout=120)

        text = r.json()["response"]

        return extract_number(text)

    except:

        return None


# ------------------------------------------------
# TASKS
# ------------------------------------------------

tasks = {

"ambiguity":[

"""
Alex enjoys solving puzzles and abstract problems.

Give probability Alex is a mathematician.
""",

"""
Jordan likes analytical thinking and logic.

Give probability Jordan works in mathematics.
"""
],


"revision":[

"""
Initially probability suspect guilty is 0.5.

New strong evidence suggests innocence.

Give updated probability suspect is guilty.
""",

"""
Drug initially believed effective with probability 0.5.

Large clinical study shows it fails.

Give updated probability drug works.
"""
],


"context_A_then_B":[

"""
Person description: analytical and intelligent.

Q1: Probability person enjoys mathematics.
Q2: Probability person is scientist.

Return probability for Q2.
"""
],


"context_B_then_A":[

"""
Person description: analytical and intelligent.

Q1: Probability person is scientist.
Q2: Probability person enjoys mathematics.

Return probability for Q2.
"""
],


# NEW PROMPTS FOR LTP TEST

"prob_A":[

"""
Person description: analytical and intelligent.

Probability person is scientist.
"""
],

"prob_A_given_not_B":[

"""
Person description: analytical and intelligent.

Person does NOT enjoy mathematics.

Probability person is scientist.
"""
],


"social":[

"""
Alice hides a key in a drawer.

Bob watches her hide it.

Probability Bob searches the drawer.
""",

"""
Sarah places a book on a table.

Tom sees her do this.

Probability Tom believes the book is on the table.
"""
],


"dynamics":[

"""
Disease initially probability 0.3.

Test result increases probability to 0.6.

Second weak test reduces confidence slightly.

Give final probability disease present.
""",

"""
Machine failure probability starts 0.4.

Sensor increases probability to 0.7.

Another reading slightly decreases confidence.

Give final probability of failure.
"""
],


"conjunction_fallacy":[

"""
Linda is 31 years old, single, outspoken, and very bright.

She majored in philosophy and was deeply concerned with discrimination and social justice.

Which probability is higher?

A: Linda is a bank teller
B: Linda is a bank teller and active in the feminist movement

Return probability for option B.
"""
],


"base_rate_neglect":[

"""
In a city 1% of people are engineers and 99% are teachers.

A randomly selected person enjoys solving math puzzles.

Probability the person is an engineer?
"""
]

}


# ------------------------------------------------
# RUN SINGLE TRIAL
# ------------------------------------------------

def run_trial(model,task,prompt):

    val = query_model(model,prompt)

    return {
        "model":model,
        "task":task,
        "value":val
    }


# ------------------------------------------------
# RUN BENCHMARK
# ------------------------------------------------

results = []

futures = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    for model in MODELS:

        for task in tasks:

            for prompt in tasks[task]:

                for _ in range(TRIALS_PER_TASK):

                    futures.append(
                        executor.submit(run_trial,model,task,prompt)
                    )


    for f in tqdm(as_completed(futures),total=len(futures)):

        r = f.result()

        if r["value"] is not None:
            results.append(r)



df = pd.DataFrame(results)

df.to_csv("llm_capability_results.csv", index=False)

print("\nSaved results to llm_capability_results.csv")


# ------------------------------------------------
# SUMMARY STATS
# ------------------------------------------------

summary = []

for (model,task),g in df.groupby(["model","task"]):

    m = g["value"].mean()
    s = sem(g["value"])

    summary.append({
        "model":model,
        "task":task,
        "mean":m,
        "sem":s
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv('summary.csv',index=False)


# ------------------------------------------------
# LAW OF TOTAL PROBABILITY TEST
# ------------------------------------------------

print("\nLAW OF TOTAL PROBABILITY VIOLATION\n")

# ltp_results = []

# for model in MODELS:

#     P_A = df[(df.model==model) & (df.task=="prob_A")]["value"].mean()

#     P_A_given_B = df[(df.model==model) & (df.task=="context_A_then_B")]["value"].mean()

#     P_A_given_not_B = df[(df.model==model) & (df.task=="prob_A_given_not_B")]["value"].mean()

#     P_B = df[(df.model==model) & (df.task=="ambiguity")]["value"].mean()

#     classical = P_A_given_B*P_B + P_A_given_not_B*(1-P_B)

#     delta = P_A - classical

#     print(model,"interference =",round(delta,4))

#     ltp_results.append({
#         "model":model,
#         "P_A":P_A,
#         "classical_prediction":classical,
#         "ltp_violation":delta
#     })


# ltp_df = pd.DataFrame(ltp_results)

# ltp_df.to_csv("ltp_violation.csv",index=False)


rows = []

for model in MODELS:

    P_A = df[(df.model==model) & (df.task=="prob_A")]["value"]

    P_A_given_B = df[(df.model==model) & (df.task=="context_A_then_B")]["value"]

    P_A_given_not_B = df[(df.model==model) & (df.task=="prob_A_given_not_B")]["value"]

    P_B = df[(df.model==model) & (df.task=="ambiguity")]["value"]

    n = min(len(P_A),len(P_A_given_B),len(P_A_given_not_B),len(P_B))

    P_A = P_A.values[:n]
    P_A_given_B = P_A_given_B.values[:n]
    P_A_given_not_B = P_A_given_not_B.values[:n]
    P_B = P_B.values[:n]

    classical = P_A_given_B*P_B + P_A_given_not_B*(1-P_B)

    delta = P_A - classical

    rows.append({
        "model":model,
        "P_A_mean":np.mean(P_A),
        "P_A_sem":sem(P_A),
        "classical_mean":np.mean(classical),
        "classical_sem":sem(classical),
        "delta_mean":np.mean(delta),
        "delta_sem":sem(delta)
    })

ltp_df = pd.DataFrame(rows)

ltp_df.to_csv("ltp_violation.csv",index=False)

# -----------------------------------------------
# Quantum Interference Angle
# -----------------------------------------------

rows = []

for model in MODELS:

    P_A = df[(df.model==model) & (df.task=="prob_A")]["value"]
    P_A_given_B = df[(df.model==model) & (df.task=="context_A_then_B")]["value"]
    P_A_given_not_B = df[(df.model==model) & (df.task=="prob_A_given_not_B")]["value"]
    P_B = df[(df.model==model) & (df.task=="ambiguity")]["value"]

    n = min(len(P_A),len(P_A_given_B),len(P_A_given_not_B),len(P_B))

    P_A = P_A.values[:n]
    P_A_given_B = P_A_given_B.values[:n]
    P_A_given_not_B = P_A_given_not_B.values[:n]
    P_B = P_B.values[:n]

    classical = P_A_given_B*P_B + P_A_given_not_B*(1-P_B)

    delta = P_A - classical

    denom = 2*np.sqrt(P_A_given_B*P_B*P_A_given_not_B*(1-P_B))

    cos_theta = delta/denom

    cos_theta = np.clip(cos_theta,-1,1)

    theta = np.arccos(cos_theta)

    rows.append({
        "model":model,
        "delta_mean":np.mean(delta),
        "delta_sem":sem(delta),
        "theta_mean":np.mean(theta),
        "theta_sem":sem(theta)
    })

results = pd.DataFrame(rows)

results.to_csv("quantum_interference_results.csv",index=False)

# ------------------------------------------------
# PLOTS
# ------------------------------------------------


plt.figure(figsize=(10,6))

sns.barplot(
    data=summary_df,
    x="task",
    y="mean",
    hue="model"
)

plt.xticks(rotation=45)

plt.title("Model Cognitive Profile")

plt.tight_layout()

plt.savefig("cognitive_profile.png")

print("\nSaved plots and LTP results.")