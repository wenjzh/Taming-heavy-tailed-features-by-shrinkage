# This script is used to analyze heavy-tailed features from CNN
# via calculating Kurtosis value and QQ plot.
#
# Author: Wenjing Zhou
# Last modified date: 06/07/2020

import scipy.stats as stats
import matplotlib.pyplot as plt

train_loader_new = torch.utils.data.DataLoader(
    dataset=dataset_train, batch_size=1, shuffle=True)

out_x_list = []
kurtosis_list = []

# model is a pretrained CNN
for m, (images, labels) in enumerate(train_loader_new):
    _, out_x = model(images.float())
    out_x_list.append(out_x)

for n in range(1000):
    out_x_point = []
    for j in range(11791):
        out_x_point.append(out_x_list[j][0][n].detach().numpy())
    k = kurtosis(out_x_point, fisher=True)
    kurtosis_list.append(k)

# Largest Kurtosis Value
max_kt = max(kurtosis_list)
print(max_kt)

# Find index of feauture with largest Kurtosis Value
point_location = kurtosis_list.index(max(kurtosis_list))
out_x_point = []

# EXtract features
for j in range(11791):
    out_x_point.append(out_x_list[j][0][point_location].detach().numpy())

# QQ plot
fig = plt.figure()
stats.probplot(out_x_point, dist="norm", plot=plt)
plt.title('The Probability Plot of No.{} output feature'.format(point_location))
plt.show()
