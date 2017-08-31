import pymongo
import cPickle

database = pymongo.MongoClient().dataSet
col = database.user_reaction_filtered

logfile = "log_kcore"
users_list_file = "data/user_dict_NYTWaPoWSJ"
items_list_file = "data/item_dict_NYTWaPoWSJ"

def slotsort(entry_dictionary):
    sorted = {}
    for ID in entry_dictionary.keys():
        value = entry_dictionary[ID]
        sorted.setdefault(entry_dictionary[ID],{})
        sorted[value][ID] = True
    return sorted


def list_del(K, entry_list, entry_list_sorted):
    entry_list_del = []
    for k in range(K):
        if k in entry_list_sorted:
             entry_list_del += entry_list_sorted[k].keys()
             for entryID in entry_list_sorted[k].keys():
                 del entry_list[entryID]
             del entry_list_sorted[k]
    return entry_list_del


def transaction_del(K = None, users_list_del = None, items_list_del = None, items_list_sorted = None, items_list = None, users_list_sorted = None, users_list = None, col_edge = None):
    if users_list_del is None:
        UDEL = False
    else:
        UDEL = True
    if UDEL:
        item_update = {}
        for userID in users_list_del:
            edge_cursor = col_edge.find({"READERID": userID}, no_cursor_timeout = True)
            for edge in edge_cursor:
                kcore = edge["KCORE"]
                if kcore != 0 and kcore <= K:
                    print "kcore", kcore
                    print "edge", edge
                    continue    # already excluded by smaller k core
                itemID = edge["POSTID"]
                # print "item to be modified", itemID
                item_update.setdefault(itemID,0)
                item_update[itemID] += 1 
                # print item_update
                edge["KCORE"] = K
                col_edge.save(edge)
        print "total item to be updated", len(item_update), "total reactions to del", sum(item_update.values())
        listUpdate(items_list, items_list_sorted, item_update)
    else:
        user_update = {}
        for itemID in items_list_del:
            edge_cursor = col_edge.find({"$and":[{"POSTID": itemID},{"KCORE": 0}]}, no_cursor_timeout = True)
            for edge in edge_cursor:
                kcore = edge["KCORE"]
                if kcore != 0 and kcore <= K:
                    print "kcore", kcore
                    print "edge", edge
                    continue    # already excluded by smaller k core
                userID = edge["READERID"]
                user_update.setdefault(userID,0)
                user_update[userID] += 1
                edge["KCORE"] = K
                col_edge.save(edge)
        print "total user to be updated", len(user_update), "total reactions to del", sum(user_update.values())
        listUpdate(users_list, users_list_sorted, user_update)


def listUpdate(entry_list, entry_list_sorted, entry_update):
    for entryID in entry_update.keys():
        old_value = entry_list[entryID]
        new_value = old_value - entry_update[entryID]
        entry_list[entryID] = new_value
        del entry_list_sorted[old_value][entryID]
        entry_list_sorted.setdefault(new_value, {})[entryID] = True


def kcoreSingle(K, users_list_sorted, items_list_sorted, users_list, items_list, col_edge):
    while True:
        users_list_del = list_del(K, users_list, users_list_sorted)
        with open(logfile, "a") as logf:
            logf.write("users to be deleted" + str(len(users_list_del)) + "\n")
        Nreaction = sum(items_list.values())
        print "Nreaction from items before", Nreaction
	transaction_del(K = K, users_list_del = users_list_del, items_list_sorted = items_list_sorted, items_list = items_list, col_edge = col_edge)
        items_list_del = list_del(K, items_list, items_list_sorted)
        with open(logfile, "a") as logf:
            logf.write("items to be deleted" + str(len(items_list_del)) + "\n")
        Nreaction = sum(items_list.values())
        print "Nreaction from items after", Nreaction
        if len(items_list_del) < 1:
            with open(logfile, "a") as logf:
                logf.write("kcore decomposition done with K = %d\n" % K)
            break
        transaction_del(K = K, items_list_del = items_list_del, users_list_sorted = users_list_sorted, users_list = users_list, col_edge = col_edge)
    return users_list, items_list, users_list_sorted, items_list_sorted


def kcore(K, users_list_file, items_list_file, col_edge, store_every_k = False):
    with open(users_list_file, "r") as f:
         users_list = cPickle.load(f)
    with open(items_list_file, "r") as f:
         items_list = cPickle.load(f)
    users_list_sorted = slotsort(users_list)
    items_list_sorted = slotsort(items_list)
    for k in range(2,K+1):
        Nreaction = sum(items_list.values())
        print "Nreaction from items before kcoreSingle", Nreaction
        kcoreSingle(k, users_list_sorted, items_list_sorted, users_list, items_list, col_edge)
        Nreaction = sum(items_list.values())
        print "Nreaction from items after kcoreSingle", Nreaction
        if store_every_k or k == K:
           with open(users_list_file[:25] + "_K" + str(k), "w") as f:
               cPickle.dump(users_list, f)
           with open(items_list_file[:25] + "_K" + str(k), "w") as f:
               cPickle.dump(items_list, f)


def RESET(K_MAX, col_edge):
    col_edge.update_many({"KCORE":{"$gt": K_MAX-1}}, {"$set": {"KCORE": 0}}, upsert = False)
    print "reset done, no k larger or equal than K_MAX"


if __name__ == "__main__":
    database = pymongo.MongoClient().dataSet
    col_edge = database.user_reaction_filtered
    # !!!!!! RESET !!!!!!!! #
    ### RESET(2, col_edge)
    #########################
    users_list_file = "data/user_dict_NYTWaPoWSJ_K10"
    items_list_file = "data/item_dict_NYTWaPoWSJ_K10"
    K = 50
    kcore(K, users_list_file, items_list_file, col_edge, store_every_k = True)
          
