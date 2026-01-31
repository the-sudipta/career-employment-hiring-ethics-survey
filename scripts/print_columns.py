import pandas as pd

PATH = "data/raw/dataset.xlsx"

def main():
    df = pd.read_excel(PATH)
    print("Rows, Cols:", df.shape)
    print("\nColumns:")
    for i, c in enumerate(df.columns, 1):
        print(f"{i:02d}. {c}")

if __name__ == "__main__":
    main()
