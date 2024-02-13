import pandas as pd
import random
from datetime import datetime, timedelta
def generate(n=1000):
    value_actions = {
        "Login":(1,100),
        "Logout":(100, 1000),
        "Edit":(1000,10000),
        "Create":(10000,100000)
    }
    business_actions = ["Login", "Logout", "Edit", "Create"]
    users =  list("USER_"+chr(i) for i in range(ord('A'), ord('Z')+1))
    data = []
    random.seed()
    for _ in range(n):
        anomaly_probability = random.randint(0,100)
        date = datetime.now() - timedelta(days=random.randint(1, 30))
        timestamp = int(date.timestamp())
        business_actions = random.sample(
            business_actions, k=len(business_actions)
        )
        action = random.choice(business_actions)
        users   =  random.sample(users, k=len(users))
        user = random.choice(users)
        intervallo = value_actions[action]
        value = random.randint(intervallo[0], intervallo[1])
        value = value * 100 if anomaly_probability < 10 else value
        status = -1 if anomaly_probability < 10 else 1
        # print(f"ACTION= {action}, user= {user},intervallo= {intervallo},value= {value}, STATUS={status}")
        # print(anomaly_probability)
        # input("ok")
        message = f"{action} generate {value} bytes performed by {user}"
        columns = ["event","machine","timestamp", "label"]
        data.append([message, "1", timestamp, status])
    return data