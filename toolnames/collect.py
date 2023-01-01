from evilai.api import create
from consts import TOOL_VERBS, TEMPLATE
import json

# print(TOOL_VERBS)

def get_tool_name(verb):
    prompt = TEMPLATE.format(verb=verb)
    response = create(engine="text-davinci-003", prompt=prompt, temperature=0.0, max_tokens=400)
    return eval(response['choices'][0]['text'])

if __name__ == '__main__':
    tool_names = {}
    for verb in TOOL_VERBS:
        tool_names[verb] = get_tool_name(verb)
        print(verb, tool_names[verb])

    with open('toolnames.json', 'w') as f:
        json.dump(tool_names, f)
