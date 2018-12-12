from pymongo import MongoClient
import pprint

client = MongoClient()
db = client.sacred

metrics = db.metrics
runs = db.runs

exp_list = [360,359,358,357,356,355,354,353,352,341,340,339,334,333,332,300,299,298,202,201,200,180,161,159,158]

results = {}
for exp in exp_list:
    res = metrics.find({"run_id": exp})
    results[exp] = dict()
    for r in res:
        results[exp][r['name']] = r['values']

results