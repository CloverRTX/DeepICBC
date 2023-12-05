from datetime import datetime

class TimeOverhead:

    def __init__(self):
        self.Cumulative_time = 0

    def timeStart(self):
        self.start = datetime.now()

    def timeEnd(self):
        self.end = datetime.now()
        self.Cumulative_time = self.Cumulative_time + (self.end - self.start).seconds

    def timeCost(self, str):
        duration = self.Cumulative_time
        day = duration // (60 * 60 * 24)
        hour = duration % (60 * 60 * 24) // (60 * 60)
        minute = duration % (60 * 60) // 60
        second = duration % 60
        print(f"=========={str}时间统计==========")
        print(f"{str} cost : {day} day(s), {hour} hour(s), {minute} minute(s), {second} second(s)")
        print("==============================")


    def timeCost_str(self):
        duration = self.Cumulative_time
        day = duration // (60 * 60 * 24)
        hour = duration % (60 * 60 * 24) // (60 * 60)
        minute = duration % (60 * 60) // 60
        second = duration % 60
        return f"{day} day(s), {hour} hour(s), {minute} minute(s), {second} second(s)"