from train_severaltimes import same_dataset

dataset_list = ["tower", "bedroom", "Celeb"]
start = 0
for dataset in dataset_list:
    same_dataset(model_name="WGPX", dataset=dataset, times=5, start_num=start, max_z=200, r_z=100, ep=400, v=10, sp=20)
    start += 5