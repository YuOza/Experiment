from train-wgp1 import train_one_time

for i in range(5):
    now_name = 'WGP1_' + str(i)
    train_one_time(nz=200, real_z=100, n_ep=400, m_name=now_name,
                   dataset_name='Celeb', value=10, space=20)
# dataset name = Celeb, tower, bedroom