import numpy as np
import ray

@ray.remote
def grouping_to_matrix(divided):

    l_train = len(divided[0][0])
    l_test = len(divided[0][1])
    M_train = np.zeros((l_train, l_train))
    M_test = np.zeros((l_test, l_train))
    for train_grouping, test_grouping in divided:
        
        # Sort the groupings to append efficiently to the kernel matrix
        temp = [(v,i) for i,v in enumerate(train_grouping)]
        temp.sort()
        train_sorted, train_indices = zip(*temp)

        temp = [(v,i) for i,v in enumerate(test_grouping)]
        temp.sort()
        test_sorted, test_indices = zip(*temp)

        for i in range(l_train):

            j = i + 1
            train_index_i = train_indices[i]
            train_sorted_i = train_sorted[i]

            while j < l_train and train_sorted_i == train_sorted[j]:
                M_train[train_index_i, train_indices[j]] += 1
                j += 1

            j = 0
            while j < l_test and train_sorted_i != test_sorted[j]:
                j += 1

            while j < l_test and train_sorted_i == test_sorted[j]:
                M_test[test_indices[j], train_index_i] += 1
                j += 1

    return M_train, M_test