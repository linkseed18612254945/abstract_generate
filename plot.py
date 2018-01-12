import matplotlib.pyplot as plt

loss_path = 'MT_save_models/title-abstract-joint-pos/loss.txt'
#loss_path = 'MT_save_models/topic-title-joint/loss.txt'
with open(loss_path) as f:
    loss = f.read().splitlines()
loss = [float(i) for i in loss]
print(loss)
plt.plot(loss)
plt.show()