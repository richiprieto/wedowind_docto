from load_data.load_dataset import process_multiple_files
from preprocessing.cleaning import clean_dataset


def main():
    base_path = "../../../aventa_rotor_icing/"
    file_list = [base_path + "Aventa_Taggenberg_01_11_2022.hdf5",
                 base_path + "Aventa_Taggenberg_04_11_2022.hdf5"]
    train_dataset = process_multiple_files(file_list)
    train_dataset = clean_dataset(train_dataset, handling_outliers='clip')
    print(train_dataset)
if __name__ == "__main__":
    main()