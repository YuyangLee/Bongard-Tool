import json
import random
from glob import glob
import argparse


def generate_tasks(functions, func_tools, tool_paths, all_tools, N_per_func=10):
    tasks = {}
    for func in functions:
        pos_tools = [tool for tool in func_tools[func] if tool in all_tools]
        neg_tools = [tool for tool in all_tools if tool not in pos_tools]
        for i in range(N_per_func):
            if len(pos_tools) < 7:
                continue
            pos_tools = random.sample(pos_tools, 7)
            neg_tools = random.sample(neg_tools, 7)
            pos_paths = [random.choice(tool_paths[tool]) for tool in pos_tools]
            neg_paths = [random.choice(tool_paths[tool]) for tool in neg_tools]
            img_paths = pos_paths[:6] + neg_paths[:6] + pos_paths[6:] + neg_paths[6:]
            tasks[f'{func}_{i}'] = img_paths
    # shuffle tasks
    tasks = list(tasks.items())
    random.shuffle(tasks)
    tasks = {task[0]: task[1] for task in tasks}
    return tasks


def main(args):
    name_path = args.name_path
    data_root = args.data_root
    
    with open(name_path, 'r') as f:
        func_tools = json.load(f)
    functions = list(func_tools.keys())
    all_tools = []
    tool_paths = {}
    for func in functions:
        tools = func_tools[func]
        for tool in tools:
            paths = glob(f'{data_root}/{tool}/*.jpg')
            if len(paths) > 0:
                tool_paths[tool] = paths
                all_tools.append(tool)
            if len(paths) == 0:
                print(f'no image for {tool}')
    all_tools = list(set(all_tools))
    print(f'functions: {len(functions)}')
    print(f'all tools: {len(all_tools)}')

    # generate bongard tasks
    random.seed(0)
    N_per_func = 100

    # Generate tasks for training and testing. Randomly split.
    print('Generate tasks for training and testing. Randomly split.')
    all_tasks = generate_tasks(functions, func_tools, tool_paths, all_tools, N_per_func=N_per_func)
    tasks = list(all_tasks.keys())
    train_tasks = tasks[:int(len(tasks)*0.8)]
    test_tasks = tasks[int(len(tasks)*0.8):]
    train_tasks = {task: all_tasks[task] for task in train_tasks}
    test_tasks = {task: all_tasks[task] for task in test_tasks}
    print(f'generate train tasks: {len(train_tasks)}')
    with open(f'{data_root}/train.json', 'w') as f:
        json.dump(train_tasks, f, indent=4)
    print(f'generate test tasks: {len(test_tasks)}')
    with open(f'{data_root}/test.json', 'w') as f:
        json.dump(test_tasks, f, indent=4)

    # Generate tasks for training and testing. 20% functions is unseen in training.
    print('Generate tasks for training and testing. 20% functions is unseen in training.')
    random.shuffle(functions)
    train_func = functions[:int(len(functions)*0.8)]
    test_func = functions[int(len(functions)*0.8):]
    train_tasks = generate_tasks(train_func, func_tools, tool_paths, all_tools, N_per_func=N_per_func)
    print(f'generate train tasks: {len(train_tasks)}')
    with open(f'{data_root}/train_func.json', 'w') as f:
        json.dump(train_tasks, f, indent=4)
    test_tasks = generate_tasks(test_func, func_tools, tool_paths, all_tools, N_per_func=N_per_func)
    print(f'generate test tasks: {len(test_tasks)}')
    with open(f'{data_root}/test_func.json', 'w') as f:
        json.dump(test_tasks, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_path', default='./toolnames/names.1.2.1.json', help='the path to the json file containing all tool names')
    parser.add_argument('--data_root', default='/home/yuliu/Dataset/Tool', help='the root directory of all classes of images')
    args = parser.parse_args()
    main(args)
    


    

