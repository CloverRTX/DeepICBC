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
        print(f"======================={str}时间统计=============================")
        print(f"{str}花费时间 : {day}天{hour}小时{minute}分钟{second}秒")
        print("=================================================================")


    def timeCost_str(self):
        duration = self.Cumulative_time
        day = duration // (60 * 60 * 24)
        hour = duration % (60 * 60 * 24) // (60 * 60)
        minute = duration % (60 * 60) // 60
        second = duration % 60
        return f"{day}天{hour}小时{minute}分钟{second}秒"