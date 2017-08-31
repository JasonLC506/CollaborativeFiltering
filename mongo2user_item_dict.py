import pymongo
import cPickle
import re


database = pymongo.MongoClient().dataSet
col_r = database.user_reaction
col_r_filtered = database.user_reaction_filtered

logfile = "user_item_dict"
users_list_file = "data/user_dict_NYTWaPoWSJ"
items_list_file = "data/item_dict_NYTWaPoWSJ"

total_users = 0
total_item = 0
query_result_r = col_r.find(no_cursor_timeout = True)

users_list = {}
items_list = {}
cnt = 0

itemID_old = None
sequence_p_post = 0

for react in query_result_r:
    # abandon Foxnews posts reactions temporarily #
    if re.match(r'15704546335', react["POSTID"]):
        continue
    userID = react["READERID"]
    itemID = react["POSTID"]
    users_list.setdefault(userID, 0)
    users_list[userID] += 1
    items_list.setdefault(itemID, 0)
    items_list[itemID] += 1
    if itemID_old is None or itemID_old != itemID:
        itemID_old = itemID
        sequence_p_post = 0
    # insert into new collection #
    col_r_filtered.insert_one(
      {
         "READERID": userID,
         "POSTID": itemID,
         "EMOTICON": react["EMOTICON"],
         "ORDERR": sequence_p_post,
         "KCORE": 1
      }
    )
    sequence_p_post += 1
    cnt += 1
    ### test ###
    # if cnt > 10000:
    #    break
    if cnt % 10000 == 0:
        print "reactions retrieved %d" % cnt
query_result_r.close()

#avg_react_p_user = 0
#for readerID in users_list.keys():
#    total_react = users_list[readerID]
#    avg_react_p_user += total_react
        
with open(users_list_file,"w") as f:
    cPickle.dump(users_list, f)
with open(items_list_file,"w") as f:
    cPickle.dump(items_list, f)
#with open(logfile, "a") as flog:
#    flog.write("++++++++++++ total_users %d +++++++++++++++++++\n" %  len(users_list))
#    flog.write("++++++++++++ average reactions per users %f +++++++++++++++\n" % (avg_react_p_user/(1.0*len(users_list))))
