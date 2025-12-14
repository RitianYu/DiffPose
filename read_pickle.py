import pickle 

with open("/nas2/home/yuhao/code/DiffPose/diffpose_deepfluoro_final_pose.pkl", "rb") as f:
    data1 = pickle.load(f)

with open("/nas2/home/yuhao/code/DiffPose/diffpose_deepfluoro_init_pose.pkl", "rb") as f:   
    data2 = pickle.load(f)

breakpoint()