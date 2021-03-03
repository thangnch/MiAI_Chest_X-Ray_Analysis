import pandas as pd
import cv2
import os
import pickle

# Đọc file CSV
csv_file = "G:\\VINAI_Chest_Xray\\train_downsampled.csv"
df = pd.read_csv(csv_file)

print(df.head())

# Lặp qua các file trong thư mục train
raw_folder = "G:\\VINAI_Chest_Xray\\train\\train"
idx = 0
file_list = []
for file in os.listdir(raw_folder):
    if file[0] != ".": # Ignore temp file
        print("Xử lý file {}- {}".format(idx, file))
        idx += 1
        # Tìm xem trong CSV có nhãn nào cho file và chỉ lọc các bản ghi có class_id khác 14
        df_find = df[(df.image_id == file[:-4]) & (df.class_id != 14)]

        # Nếu tìm thấy bản ghi phù hợp
        if len(df_find) > 0:
            # Đọc file ảnh để lấy kích thước
            raw_image = cv2.imread(os.path.join(raw_folder, file), 0)
            image_width, image_height = raw_image.shape[1], raw_image.shape[0]

            labels = []

            # Lặp qua từng bản ghi trong df_find để tính toán
            for index, row in df_find.iterrows():

                # Tính tọa độ tâm và kích thước theo pixels
                box_width = row[6] - row[4]
                box_height = row[7] - row[5]
                box_center_x = (row[6] + row[4]) / 2
                box_center_y = (row[7] + row[5]) / 2

                # Thực hiện chuẩn hóa
                box_width_normalized = box_width / image_width
                box_height_normalized = box_height / image_height
                box_center_x_normalized = box_center_x / image_width
                box_center_y_normalized = box_center_y / image_height

                # Ghi thông tin nhãn vào list
                labels.append([row[2], box_center_x_normalized, box_center_y_normalized, box_width_normalized, box_height_normalized])

            # Lặp qua list và ghi vào file txt
            txt_file = file[:-4] + ".txt"
            with open(os.path.join(raw_folder, txt_file),'w') as f:
                for label in labels:
                    f.write('{} {} {} {} {}\n'.format(label[0],label[1],label[2],label[3],label[4]))
            print("Ghi xong nhãn ", txt_file)
            file_list.append(file)
            print("Số file có nhãn = ", len(file_list))

# Ghi file_list vào file pickle
with open('file_list.pkl','wb') as f:
    pickle.dump(file_list,f)