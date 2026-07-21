import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build Cityscapes video evaluation list files')
    parser.add_argument('mode', type=str, help='split name (e.g. val)')
    parser.add_argument('num_frames', type=int, help='temporal lookback (N in val_sequence_N_frame.lst)')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_path_min = 'leftImg8bit/' + args.mode + '/'
    label_path_min = 'gtFine/' + args.mode + '/'
    data_path = 'data/cityscapes/leftImg8bit/' + args.mode + '/'
    label_path = 'data/cityscapes/gtFine/' + args.mode + '/'

    f = open(f'{args.mode}_sequence_{args.num_frames}_frame.lst', 'a')
    i = 0
    flag = False
    for city in sorted(os.listdir(data_path)):
        all_labels = sorted(os.listdir(os.path.join(label_path, city)))
        all_jsons = [file for file in all_labels if file.endswith('.json')]
        img_name_inc = ['_'.join(name.split('_')[:3]) for name in all_jsons]
        for idx, img in enumerate(sorted(os.listdir(os.path.join(data_path, city)))):
            i += 1
            if '_'.join(img.split('_')[:3]) in img_name_inc:
                f.write(data_path_min + img.split('_')[0] + '/' + img + ' ' + label_path_min + img.split('_')[0] + '/' + '_'.join(img.split('_')[:3]) + '_gtFine_labelIds.png' + '\n')
                flag = True
                i_prev = i
            elif flag and i < i_prev + 10:
                continue
            elif flag:
                flag = False
            else:
                f.write(data_path_min + img.split('_')[0] + '/' + img + '\n')
    f.close()


if __name__ == '__main__':
    main()
