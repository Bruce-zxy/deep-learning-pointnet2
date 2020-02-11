import numpy as np

bsize=3

batch_label=[0,1,2]
pred_val=[2,2,1]

print(len(batch_label))
print(len(pred_val))

total_seen_class = [0 for _ in range(3)]
total_correct_class = [0 for _ in range(3)]

for i in range(0, bsize):
    print('INNER: ', i)
    l = batch_label[i]
    total_seen_class[l] += 1
    total_correct_class[l] += (pred_val[i] == l)

print(total_seen_class)
print(total_correct_class)

print(np.mean(np.array(total_correct_class) /
              np.array(total_seen_class, dtype=np.float)))
