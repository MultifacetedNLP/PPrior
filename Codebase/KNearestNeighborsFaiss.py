import numpy as np
import faiss
from faiss.contrib.ondisk import merge_ondisk


class FaissKNeighbors:
    def __init__(self, d, path, batch_size, gpu):
        if gpu == True:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, faiss.index_factory(d, "IVF4000,Flat"))
        else:
            self.index = faiss.index_factory(d, "IVF4000,Flat")
        self.y = np.array([], dtype=np.int64)
        self.path = path
        self.batch_num = 0
        self.batch_size = batch_size
        self.class_size = np.array([617115, 320318, 139776, 67629, 30933])

    def fit(self, X, y):
        if self.batch_num == 0:
            self.index.train(X.astype(np.float32))
            faiss.write_index(self.index, f'{self.path}/trained.index')

        index = faiss.read_index(f'{self.path}/trained.index')
        size = X.shape[0]
        index.add_with_ids(X.astype(np.float32),
                           np.arange(self.batch_num * self.batch_size, self.batch_num * self.batch_size + size))
        faiss.write_index(index, f'{self.path}/block_{self.batch_num}.index')

        self.batch_num = self.batch_num + 1
        self.y = np.concatenate((self.y, y))

    def predict(self, X, k=5, nprobe=80):
        self.index.nprobe = nprobe
        distances, indices = self.index.search(X.astype(np.float32), k=k)
        votes = self.y[indices]
        predictions = [np.argmax((np.bincount(x, minlength=5) / self.class_size)) for x in votes]
        return predictions

    def predict_radius(self, X, radius=1.5, nprobe=80):
        self.index.nprobe = nprobe
        lims, D, I = self.index.range_search(X.astype(np.float32), radius)
        predictions = []

        for index in range(X.shape[0]):
            votes = self.y[I[lims[index]:lims[index + 1]]]
            if votes.size == 0:
                votes = np.array([0])
            prediction = np.argmax((np.bincount(votes, minlength=5)))
            predictions.append(prediction)

        return predictions

    def predict_weighting_neighbour(self, X, k=5, nprobe=80):
        self.index.nprobe = nprobe
        distances, indices = self.index.search(X.astype(np.float32), k=k)
        weights = 1 / (distances + 1e-10)
        votes = self.y[indices]
        predictions = []
        for index in range(votes.shape[0]):
            vote = votes[index]
            weight = weights[index]

            one_hot = np.zeros((vote.size, 5))
            one_hot[np.arange(vote.size), vote] = 1

            prediction = np.argmax(weight.dot(one_hot))
            predictions.append(prediction)

        return predictions

    def save(self):
        index = faiss.read_index(f'{self.path}/trained.index')
        block_fnames = [f'{self.path}/block_{b}.index' for b in range(self.batch_num)]
        merge_ondisk(index, block_fnames, f'{self.path}/merged_index.ivfdata')
        faiss.write_index(index, f'{self.path}/populated.index')
        np.save(f'{self.path}/trained.npy', self.y)

    def load(self, gpu):
        if gpu == True:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, faiss.read_index(f'{self.path}/populated.index',
                                                                         faiss.IO_FLAG_ONDISK_SAME_DIR))
        else:
            self.index = faiss.read_index(f'{self.path}/populated.index', faiss.IO_FLAG_ONDISK_SAME_DIR)

        self.y = np.load(f'{self.path}/trained.npy')


def train_knn(kneighbor, model, train_dataset, batch_size):
    def fit_on_dataset(batch):
        kneighbor.fit(model.encode(batch['text'], convert_to_numpy=True), batch['label'])

    train_dataset.map(fit_on_dataset, batched=True, batch_size=batch_size)
    kneighbor.save()


def search_knn_first(kneighbor, model, test_dataset, K, nprobe, batch_size):
    kneighbor.load(gpu=True)

    def predict_on_dataset(batch):
        return {"predicted_label": kneighbor.predict(model.encode(batch['text']), K, nprobe)}

    result = test_dataset.map(predict_on_dataset, batched=True, batch_size=batch_size, remove_columns=['text'])
    return result


def search_knn_second(kneighbor, model, test_dataset, K, nprobe, batch_size):
    kneighbor.load(gpu=True)

    def predict_on_dataset(batch):
        return {"predicted_label": kneighbor.predict_weighting_neighbour(model.encode(batch['text']), K, nprobe)}

    result = test_dataset.map(predict_on_dataset, batched=True, batch_size=batch_size, remove_columns=['text'])
    return result


def search_knn_third(kneighbor, model, test_dataset, radius, nprobe, batch_size):
    kneighbor.load(gpu=False)

    def predict_on_dataset(batch):
        return {"predicted_label": kneighbor.predict_radius(model.encode(batch['text']), radius, nprobe)}

    result = test_dataset.map(predict_on_dataset, batched=True, batch_size=batch_size, remove_columns=['text'])
    return result
