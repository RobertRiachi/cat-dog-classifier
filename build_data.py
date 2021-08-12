# Transforming https://www.kaggle.com/c/dogs-vs-cats/data data into usable CSV

import os

if __name__ == "__main__":
    root_dir = "/mnt/c/Users/Robert/Downloads/cats-vs-dogs/train"

    data_dict = {}
    for file_name in os.listdir(root_dir):
        animal_type = file_name.split(".")[0]
        # Populate 0 for cat and 1 for dog
        data_dict[file_name] = 0 if animal_type == "cat" else 1

    csv_file = "/mnt/c/Users/Robert/Downloads/cats-vs-dogs/train_data.csv"

    print(len(data_dict))

    with open(csv_file, 'w') as f:
        for key in data_dict.keys():
            f.write(key+","+str(data_dict[key])+"\n")

