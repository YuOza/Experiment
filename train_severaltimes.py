from train-wgp1 import train_one_time

def same_dataset(model_name, dataset, times, start_num, max_z, r_z, ep, v, sp)
    end = start_num + times
    for i in range(start_num, end):
        now_name = model_name + "_" + str(i)
        train_one_time(nz=max_z, real_z=r_z, n_ep=ep, m_name=now_name,
                    dataset_name=dataset, value=v, space=sp)
# dataset name = tower, bedroom, Celeb
if __name__ == '__main__':
    same_dataset(model_name="WGPX", dataset="tower", times=5, start_num=0, max_z=200, r_z=100, ep=400, v=10, sp=20)