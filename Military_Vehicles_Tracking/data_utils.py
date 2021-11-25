import os


def read_txt_file(path):
    with open(path, 'r') as reader:
        txt_string = reader.read()
    return txt_string


def write_txt_file(path, txt):
    with open(path, 'w') as f:
        f.write(txt)


def replace_class_number(label_path, from_class, to_class):
    rows = read_txt_file(label_path).split('\n')

    new_rows = []
    for row in rows:
        row_split = row.split(' ')
        if row_split[0] == f'{from_class}':
            row_split[0] = f'{to_class}'
            new_row = " ".join(row_split)
            new_rows.append(new_row)
        else:
            new_rows.append(row)

    write_txt_file(path=label_path, txt="\n".join(new_rows))


def merge_annotation_files(label_path1, label_path2, target_path):
    annotated_labels1 = read_txt_file(label_path1)
    annotated_labels2 = read_txt_file(label_path2)
    combined_annotation = "\n".join([annotated_labels1, annotated_labels2])
    write_txt_file(path=target_path, txt=combined_annotation)


class LabelsYOLO:
    def __init__(self, train_labels_dir, val_labels_dir):
        self.train_labels_dir = train_labels_dir
        self.train_labels_files = sorted(os.listdir(train_labels_dir))
        self.val_labels_dir = val_labels_dir
        self.val_labels_files = sorted(os.listdir(val_labels_dir))

