from libraries.functions import combine_feature_data


def separate_data(calf_split_info, feature_data):
    # all calve ids
    all_calves = calf_split_info['all_calves']

    # test calve ids
    test_calves = calf_split_info['test_set']

    # train and validation calve ids
    train_validation_calves = [calf for calf in all_calves if calf not in test_calves]

    # validation calves
    validation_sets = calf_split_info['validation_sets']
    
    # Train / Validation separation
    calf_data_index = {} # keep track of the data indexes per calf
    index = 0

    X_train = []
    y_train = []
    for calf in train_validation_calves:
        sub_X_train, sub_Y_train = combine_feature_data(feature_data[calf])
        X_train.extend(sub_X_train)
        y_train.extend(sub_Y_train)

        # keeping track of train data indexes per calf
        sub_index = []
        for i in range(index, len(sub_X_train)+index):
            sub_index.append(i)
        index = index + len(sub_X_train)

        calf_data_index[calf] = sub_index
        
    train_index_sets = []
    vaildation_index_sets = []

    for validation_set in validation_sets:
        train_set = [x for x in train_validation_calves if x not in validation_set]

        X_train_indexs = []
        for calf in train_set:
            X_train_indexs.extend(calf_data_index[calf])

        X_validation_indexes = []
        for calf in validation_set:
            X_validation_indexes.extend(calf_data_index[calf])

        train_index_sets.append(X_train_indexs)
        vaildation_index_sets.append(X_validation_indexes)
        
    # Test data separation
    X_test = []
    y_test = []
    for calf in test_calves:
        sub_X_test, sub_Y_test = combine_feature_data(feature_data[calf])
        X_test.extend(sub_X_test)
        y_test.extend(sub_Y_test)
        
        
    return X_train, y_train, X_test, y_test, train_index_sets, vaildation_index_sets