import os

if __name__ == "__main__":
    # 定义要删除的文件夹路径
    folder_path = "/media/unitx/预训练模型数据_2T-2/预训练数据_3F/k3_03"

    type_ids = set()
    for filename in os.listdir(folder_path):
        type_id = filename.split("-")[0]
        type_ids.add(type_id)
    
    print("Unique type_ids found in the folder:")
    for type_id in type_ids:
        print(type_id)