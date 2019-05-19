
from pymongo import MongoClient
from bson.binary import Binary
import _pickle
import numpy as np
import random

from sum_tree import SumTree
per_a = 0.5
per_b = 0.4
per_b_per_step = 0.0001
per_e = 0.001
per_max_priority = 1.0


class MemoryDB:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.00001
    a = 0.7
    beta = 0.4
    beta_increment_per_sampling = 0.00001
    capacity = 100000
    max_priority = 1.0

    def __init__(self, host_name, db_name, collection_name):
        self.host_name = host_name
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MongoClient(host_name, 27017)
        self.db = self.client[db_name]
        self.replay_memory_collection = self.db[collection_name]
        self.sum_tree = SumTree(self.capacity)
        memory_priorities = self.replay_memory_collection.find({}, {
                                                               "priority": 1})
        for memory_priority in memory_priorities:
            self.sum_tree.add(memory_priority["priority"], {
                              "_id": memory_priority["_id"]})

    def retrieve_by_id(self, id):
        db_experiences = self.replay_memory_collection.find({"_id": id})
        return {**_pickle.loads(db_experiences[0]['binary'], encoding='latin1'), "_id": id, "priority": db_experiences[0]["priority"]}

    def _get_priority(self, error):
        return min((self.max_priority, (error + self.e) ** self.a))

    def add(self, error, experience):
        p = self._get_priority(error)
        experience_to_save = {}
        experience_to_save["terminal"] = experience["terminal"]
        experience_to_save["action_index"] = experience["action_index"]
        experience_to_save["actual_reward"] = experience["actual_reward"]
        experience_to_save["priority"] = p
        experience_to_save["binary"] = _pickle.dumps(experience)
        id = self.replay_memory_collection.insert(experience_to_save)

        entry_overwritten = self.sum_tree.add(p, {"_id": id})

        if entry_overwritten != 0:
            self.replay_memory_collection.delete_one(
                {"_id": entry_overwritten["_id"]})

    def add_batch(self, experiences):
        for experience in experiences:
            self.add(self.max_priority, experience)

    def update(self, index, error, experience):
        p = self._get_priority(error)
        self.replay_memory_collection.update_one(
            {"_id": experience["_id"]}, {"$set": {"priority": p}})
        print("index: ", index, "old priority:", experience["priority"] ,"new priority:", p)
        self.sum_tree.update(index, p)

    def update_batch(self, indexes, errors, experiences):
        for index, error, experience in zip(indexes, errors, experiences):
            self.update(index, error, experience)

    def get_experiences_size(self):
        return self.replay_memory_collection.count()

    def sample(self, n):
        
        batch = []
        idxs = []
        segment = self.sum_tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.sum_tree.get(s)
            experience = self.retrieve_by_id(data["_id"])
            priorities.append(p)
            
            batch.append(experience)
            idxs.append(idx)

        sampling_probabilities = priorities / self.sum_tree.total()
        is_weight = np.power(self.sum_tree.n_entries *
                             sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight
