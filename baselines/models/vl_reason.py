import os
import json
import clip
import torch
import openai
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

openai.api_key = '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'


def InitChatGPT():
    start_prompt = "Let us play a game named Bongard-Tool. " +\
                   "I will give you 6 tools that can perform a function, " +\
                   "and 6 tools that can not perform that function. " +\
                   "Then I will ask you whether another tool could perform that function. " +\
                   "All you need to do is guess the possible function, and tell me yes or no. " +\
                   "For example, there is a function that can be performed by these tools: " +\
                   "['Styrofoam', 'Stone', 'Balsa-Wood', 'Cardboard', 'Wooden-Block', 'Clay'], " +\
                   "and this function can not be performed by these tools: " +\
                   "['Anchors', 'Padlock', 'Smartphone', 'Hacksaw', 'Tile-Saw', 'Scrub-Brush']. " +\
                   "What function it may be? Can Plaster be used to perform this function? " +\
                   "Can Robot-Arm be used to perform this function? " +\
                   "You just need to answer: Possible functions: carving, crafting, or sculpture. Yes, No."
    print(start_prompt)
    return start_prompt
    # response = openai.Completion.create(
    #     model="text-davinci-003",
    #     prompt=start_prompt,
    #     temperature=0,
    #     max_tokens=100,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    # )
    # print(response['choices'][0]['text'].strip())


def ReasonByChatGPT(x):
    # x: list of tool classes, 6 pos + 6 neg + 1 pos + 1 neg
    prompt = f"There is a function that can be performed by these tools: {x[:6]}, " +\
             f"and this function cannot be performed by these tools: {x[6:12]}. " +\
             f"What function it may be? " +\
             f"Can {x[12]} be used to perform this function? " +\
             f"Can {x[13]} be used to perform this function? "
    return prompt
    # print(prompt)

    # response = openai.Completion.create(
    #     model="text-davinci-003",
    #     prompt=prompt,
    #     temperature=0,
    #     max_tokens=100,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    # )
    # print(response['choices'][0]['text'].strip())


def ReasonByGPT(x):
    # x: list of tool classes, 6 pos + 6 neg + 1 pos + 1 neg
    prompt = f"There is a function that can be performed by these tools: {x[:6]}, " +\
             f"and this function cannot be performed by these tools: {x[6:12]}. " +\
             f"What function it may be? " +\
             f"Can {x[12]} be used to perform this function? " +\
             f"Can {x[13]} be used to perform this function? "
    print(prompt)

    # response = openai.Completion.create(
    #     engine="text-davinci-003",
    #     prompt=prompt,
    #     temperature=0.,
    #     max_tokens=200,
    #     frequency_penalty=0.,
    # )
    # print(response['choices'][0]['text'].strip())


class Classifier(nn.Module):
    def __init__(self, device, name_path):
        super().__init__()
        self.device = device
        self.clip, self.process = clip.load('ViT-B/32', device=device)
        self.tools, self.query = self.get_query(name_path)
        

    def forward(self, img_paths):
        # img_paths: list of image paths
        imgs = []
        for img in img_paths:
            x = Image.open(img)
            x = self.process(x).to(self.device)
            imgs.append(x)
        imgs = torch.stack(imgs)
        img_features = self.clip.encode_image(imgs)
        img_features = img_features / img_features.norm(dim=1, keepdim=True)

        logits = img_features @ self.query.T
        pred = logits.argmax(dim=-1)
        pred = [self.tools[i] for i in pred]
        return pred 


    def get_query(self, name_path):
        # name_path: path to the name file
        with open(name_path, 'r') as f:
            func_tools = json.load(f)
        functions = list(func_tools.keys())
        all_tools = []
        for func in functions:
            all_tools.extend(func_tools[func])
        all_tools = list(set(all_tools))
        query = [f'a photo of {tool}' for tool in all_tools]
        query = clip.tokenize(query).to(self.device)
        query = self.clip.encode_text(query)
        query = query / query.norm(dim=-1, keepdim=True)
        return all_tools, query


'''test'''
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # model = Classifier('cuda', '/home/yuliu/Projects/Bongard-Tool/toolnames/names.1.2.1.json')
    split = 'test_func'
    data_root = '/home/yuliu/Dataset/Tool'
    # test
    with open(f'{data_root}/{split}.json', 'r') as f:
        files = json.load(f)
    keys = list(files.keys())
    img_files = [files[key] for key in keys]
    concepts = keys
    idx = 1000
    # accs = []
    # model.eval()
    start_prompt = InitChatGPT()
    prompts = [start_prompt]
    with torch.no_grad():
        for i in tqdm(range(len(img_files))):
            # if i == idx:
            #     pass
            img_paths = img_files[i]
            concept = concepts[i]
            print('concept:', concept)
            # pred = model(img_paths)
            ground_truth = [f.split('/')[-2] for f in img_paths]
            prompt = ReasonByChatGPT(ground_truth)
            prompts.append(prompt)
            # acc = sum([1 if p == g else 0 for p, g in zip(pred, ground_truth)]) / len(pred)
            # print(acc)
            # accs.append(acc)
    with open('prompts.txt', 'w') as f:
        f.write('\n'.join(prompts))
    # print(sum(accs) / len(accs))
    # print('pred:', pred)
    print('gt:', ground_truth)
    # visualize
    # from torchvision import transforms
    # import torchvision
    # T1 = transforms.ToTensor()
    # T2 = transforms.ToPILImage()
    # imgs = []
    # for path in img_paths:
    #     img = Image.open(path)
    #     img = T1(img)
    #     imgs.append(img)
    # imgs = torch.stack(imgs)
    # grid = torchvision.utils.make_grid(imgs, nrow=6)
    # grid = T2(grid)
    # grid.save('test.png')
        

        
