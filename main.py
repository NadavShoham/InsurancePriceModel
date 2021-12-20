from insurance_data import InsuranceData


def main():
    path = "insurData.csv"
    data = InsuranceData(path)
    data.build_model()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
