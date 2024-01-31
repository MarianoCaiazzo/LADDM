import pandas as pd
import random
from datetime import datetime, timedelta
def generate(n=1000):
    business_actions = ["Login", "Logout", "Edit", "Create"]
    value_action = [random.randint(10,200), random.randint(1,20), random.randint(1000,20000), random.randint(10000,20000)]
    users =  list("USER_"+chr(i) for i in range(ord('A'), ord('Z')+1))
    data = []
    for _ in range(n):
        anomaly_probability = random.randint(0,100)
        date = datetime.now() - timedelta(days=random.randint(1, 30))
        timestamp = int(date.timestamp())
        action = random.choice(business_actions)
        user = random.choice(users)
        value = value_action[business_actions.index(action)]
        value = value * 50 if anomaly_probability < 5 else value
        status = -1 if anomaly_probability < 5 else 1
        message = f"{action} generate {value} bytes performed by {user}"
        columns = ["event","machine","timestamp", "label"]
        data.append([message, "1", timestamp, status])
        df = pd.DataFrame(data, columns=columns)
    return df