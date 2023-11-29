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

def info_log(ep, loss, loss1, loss2, loss3, loss4):
    with open(r"Log\info_log.txt", "a+", encoding='utf-8') as f:
        f.write("========================info_log========================\n")
        f.write("————————————info————————————\n")
        f.write(f"The ep is {ep}\n")
        f.write(f"total_loss : {loss}\n")
        f.write(f"loss1 : {loss1}\n")
        f.write(f"loss2 : {loss2}\n")
        f.write(f"loss3 : {loss3}\n")
        f.write(f"loss4 : {loss4}\n")
