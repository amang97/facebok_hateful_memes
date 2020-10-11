from pathlib import Path

from .utilities.data_handler import DataHandler

def main():
    data_path = Path('./data')
    dh = DataHandler(data_path)
    data = dh.load_all_data()
    tr, v, t = dh.load_given_data()
    dh.compute_data_analytics(data)
    print(dh.da)
    return 0

if __name__ == "__main__":
    main()
