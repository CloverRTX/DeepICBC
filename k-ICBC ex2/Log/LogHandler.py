def train_record_log(loop, clock, str = ''):
    with open(r"Log\training_log.txt", "a+", encoding='utf-8') as f:
        f.write("========================training_log========================\n")
        f.write("————————————start training————————————\n")
        f.write(str + "\n")
        f.write(f"The loop is {loop}\n")
        f.write(f"start time is: {clock.start}\n")
        f.write(f"end time is: {clock.end}\n")
        f.write(f"cost : {clock.timeCost_str()}\n")


def verify_record_log(loop, clock, str = ''):
    with open(r"Log\verify_log.txt", "a+", encoding='utf-8') as f:
        f.write("========================verify_log========================\n")
        f.write("————————————start verify————————————\n")
        f.write(str + "\n")
        f.write(f"The loop is {loop}\n")
        f.write(f"start time is: {clock.start}\n")
        f.write(f"end time is: {clock.end}\n")
        f.write(f"cost : {clock.timeCost_str()}\n")