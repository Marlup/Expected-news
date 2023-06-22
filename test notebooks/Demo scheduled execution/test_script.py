from datetime import datetime
def show_test(date):
    with open(f"Test_sch_file_{date}.txt", "w") as f:
        f.write("Hello. This is a scheduled script at {date}!")
    print("File, Test_sch_file created!")

if __name__ == "__main__":
    date = str(datetime.today()) \
              .replace("-", "_") \
              .replace(":", " ") \
              .split(".")[0]
    show_test(date)