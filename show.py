import pickle

with open("fixed_pricer/best_bots.pkl", "rb") as f:
    a = pickle.load(f)
    print(a)
