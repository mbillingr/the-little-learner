module IrisDataset
export iris_test_xs, iris_test_ys, iris_train_xs, iris_train_ys

import MLDatasets.Iris
import DataFrames
using ..Schemish
using ..Tensors

data = Iris()

# we pick the first 5 of each species for testing
test_idx = [1:5..., 51:55..., 101:105...]
train_idx = [6:50..., 56:100..., 106:150...]

test_features = data.features[test_idx, :]
train_features = data.features[train_idx, :]
test_targets = data.targets[test_idx, :]
train_targets = data.targets[train_idx, :]

function to_tensor(df)
    let df = permutedims(df)
        tensor([df[:, c] for c in 1:DataFrames.ncol(df)])
    end
end

one_hot_encode(label) =
    if label == "Iris-setosa"
        tensor([1.0, 0.0, 0.0])
    elseif label == "Iris-versicolor"
        tensor([0.0, 1.0, 0.0])
    elseif label == "Iris-virginica"
        tensor([0.0, 0.0, 1.0])
    end

to_one_hot_tensor(df) =
    tensor(map(one_hot_encode, df[:, :class]))

iris_test_xs = to_tensor(test_features)
iris_test_ys = to_one_hot_tensor(test_targets)

iris_train_xs = to_tensor(train_features)
iris_train_ys = to_one_hot_tensor(train_targets)

end