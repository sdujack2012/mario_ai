
from pymongo import MongoClient
from bson.binary import Binary
import _pickle


class MemeryDB:
    def __init__(self, host_name, db_name, collection_name):
        self.host_name = host_name
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MongoClient(host_name, 27017)
        self.db = self.client[db_name]
        self.replay_memory_collection = self.db[collection_name]

    def save_experiences(self, experiences):
        for experience in experiences:
            experience_to_save = {}
            experience_to_save["terminal"] = experience["terminal"]
            experience_to_save["action_index"] = experience["action_index"]
            experience_to_save["actual_reward"] = experience["actual_reward"]
            experience_to_save["binary"] = _pickle.dumps(experience)
            self.replay_memory_collection.insert(experience_to_save)
        return self.replay_memory_collection

    def get_sampled_experiences(self, number_of_samples):
        db_experiences = self.replay_memory_collection.aggregate([{'$sample': {"size": number_of_samples}}], allowDiskUse=True)
        return [_pickle.loads(db_experience['binary'], encoding='latin1') for db_experience in db_experiences]
    
    def get_experiences_size(self):
        return self.replay_memory_collection.count()        