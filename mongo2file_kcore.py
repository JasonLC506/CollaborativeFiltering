import pymongo

# emoticon_list = ["Like","Love","Sad","Wow","Haha","Angry"]
EMOTICON_LIST = {"LIKE":0, "LOVE":1, "SAD":2, "WOW":3, "HAHA":4, "ANGRY":5}

col = pymongo.MongoClient().dataSet.user_reaction_filtered

K = 10 # include kcore k = 10

filename = "data/reaction_NYTWaPoWSJ_K%d" % K
cnt = 0

BATCH = 10000
batch_cnt = 0
batch = []

cursor = col.find({"KCORE": {"$gt": K}}, no_cursor_timeout = True)
for entry in cursor:
    reaction = {}
    reaction["READERID"] = entry["READERID"]
    reaction["POSTID"] = entry["POSTID"]
    reaction["KCORE"] = entry["KCORE"]
    reaction["ORDERR"] = entry["ORDERR"]
    reaction["EMOTICON"] = EMOTICON_LIST[entry["EMOTICON"]]
    cnt += 1
    batch.append(reaction)
    batch_cnt += 1
    if batch_cnt >= BATCH:
        with open(filename, "a") as f:
            for item in batch:
                f.write(str(item) + "\n")
        del batch
        batch = []
        batch_cnt = 0

with open(filename, "a") as f:
    for item in batch:
        f.write(str(item) + "\n")


cursor.close()
