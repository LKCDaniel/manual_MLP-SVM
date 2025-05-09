import pickle

with open('lfw.pkl','rb') as file:
    flw_people = pickle.load(file)

# ['data', 'images', 'target', 'target_names', 'DESCR']

# print(flw_people['DESCR'])
print(flw_people['DESCR'])
#shape