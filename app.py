import json
import shutil
from PIL import Image
from io import BytesIO
from pathlib import Path
from base64 import b64decode
from argparse import ArgumentParser
from time import localtime, strftime


# TODO: add full v2 support
current_path = Path(__name__).parent


def create_dataset_backup(dataset_path, dataset_backup_path):
    user_input = input(
        f"Do you need to create backup of '{dataset_path}'? (Type Yes or No, please) \n").lower()
    create_copy = user_input == 'yes' or user_input == 'y' or user_input == '1'

    if not create_copy:
        return

    print(f'Creating copy of {dataset_path}, this may take a while...')
    shutil.copytree(dataset_path, dataset_backup_path)


def save_plot(cell, index, name):
    plot_output = ''
    for output in cell['outputs']:
        data = output.get('data')
        if not data:
            continue
        plot_output = data.get('image/png')
        if not plot_output:
            continue
        else:
            break
    if not plot_output:
        return False

    image_data = plot_output
    plot_name = f'{name}.jpg' if index == 0 else f'{name}_{index}.jpg'

    with open(plot_name, 'wb') as f:
        f.write(b64decode(image_data))

    return True


def work_on_dataset(s, dataset_index):
    dataset_path = s.split('=')[1].strip()
    if dataset_path.startswith('Path('):
        dataset_path = str(eval(dataset_path))
    dir_name = 'dataset_backup' if dataset_index == 0 else f'dataset_{dataset_index}_backup'
    create_dataset_backup(dataset_path, project_path/dir_name)


def extract(nb_data, project_path):
    cells = nb_data['cells']

    output_data = {
        'transforms': [],
        'validation_parts': [],
        'image_sizes': [],
        'resize_methods': [],
        'bs': [],
        # 'clean_csv': '',
        'pretrained_models': [],
        'learner_types': [],
        # 'loss_fns': [],
        'min_gradients': [],
        'min_losses': [],
        'lr': [],
        'best_models': [],
    }

    learner_plot_index, top_losses_index, confusion_index, dataset_index = 0, 0, 0, 0

    # TODO: rewrite better
    for cell in cells:
        if not cell['cell_type'] == 'code':
            continue

        for s in cell['source']:
            if s.strip().startswith('#'):
                continue

            s = s.replace(' ', '')

            # DATASET COPYING
            if 'path=' in s.replace(' ', ''):
                work_on_dataset(s, dataset_index)
                dataset_index += 1

            # TRANSFORMS
            if '=get_transforms' in s:
                output_data['transforms'].append(s)

            # IMAGE SIZE
            if 'size=' in s and not 'figsize' in s and not 'batch_size' in s:
                output_data['image_sizes'].append(
                    s.split('size=')[1].rstrip(')\n').split(',')[0])

            # VALIDATION PART
            if 'split_by_rand_pct' in s:
                output_data['validation_parts'].append(
                    s.split('split_by_rand_pct(')[1].rstrip(')\n').split(',')[0])

            # RESIZE METHOD
            if 'resize_method=' in s:
                output_data['resize_methods'].append(
                    s.split('resize_method=')[1].rstrip(')\n').split(',')[0])

            # BS
            if 'bs=' in s:
                output_data['bs'].append(
                    s.split('bs=')[1].rstrip(')\n').split(',')[0])

            # PRETRAINED MODELS, LEARNER TYPES
            if '_learner' in s:
                params = s.split(',')
                if len(params) > 1:
                    output_data['pretrained_models'].append(params[1].strip())
                output_data['learner_types'].append(
                    s.split('=')[1].split('(')[0].strip())

            # LR
            if 'fit_one_cycle' in s or 'fine_tune' in s:
                params = s.split(',')
                if len(params) > 1:  # not only epoch
                    lr = params[1].strip()
                    if '(' in lr and ')' not in lr:
                        lr += ', ' + params[2].strip()
                    output_data['lr'].append(lr)

                    for p in params:
                        if 'name=' in p:
                            output_data['best_models'].append(
                                p.split("name='")[1].rstrip(",\')]"))

            # IMAGES
            if '.recorder.plot' in s:
                success = save_plot(cell, learner_plot_index,
                                    project_path/'images/learner_plot')
                learner_plot_index += 1 if success else learner_plot_index

                for output in cell['outputs']:
                    if output.get('name') == 'stdout':
                        texts = output['text']
                        for t in texts:
                            if 'numerical gradient' in t:
                                output_data['min_gradients'].append(
                                    t.split(':')[1].strip(' \n'))
                            if 'loss divided' in t:
                                output_data['min_losses'].append(
                                    t.split(':')[1].strip(' \n'))

            if '.plot_top_losses' in s:
                success = save_plot(cell, top_losses_index,
                                    project_path/'images/top_losses')
                top_losses_index += 1 if success else top_losses_index

            if '.plot_confusion_matrix' in s:
                success = save_plot(cell, confusion_index,
                                    project_path/'images/confusion_matrix')
                confusion_index += 1 if success else confusion_index

    return output_data


if __name__ == "__main__":
    parser = ArgumentParser(
        'Experiment tracker, saves all info of your notebook run so you could reproduce your results')
    parser.add_argument(
        'source_path', help='Path to .ipynb file with experiment')
    parser.add_argument('-d', '--dest_path', default=f'{current_path}',
                        help='Path to experiment data backup, default is %(default)s')
    parser.add_argument('-s', '--save_notebook', action='store_true',
                        help='Saves notebook inside project directory')

    args = parser.parse_args()
    source_path = Path(args.source_path)
    dest_path = Path(args.dest_path)
    save_notebook = args.save_notebook

    project_name = f"{source_path.stem}_{strftime('%H:%M_%d.%m.%Y', localtime())}"

    project_path = dest_path/project_name
    project_path.mkdir(parents=True, exist_ok=True)

    images_path = project_path/'images'
    images_path.mkdir(parents=True, exist_ok=True)
    dataset_path = project_path/'dataset_backup'

    if save_notebook:
        shutil.copy(source_path, project_path/source_path.name)

    with open(source_path, 'r') as f:
        nb_data = json.load(f)

    output = extract(nb_data, project_path)

    with open(project_path/'experiment_data.json', 'w') as f:
        json.dump(output, f)

    print(f'Everything saved into {project_path}')
