import json
import sys
from http import HTTPStatus

from dashscope import Assistants, Messages, Runs, Threads
from dotenv import load_dotenv

load_dotenv()

def create_assistant():
    # create assistant with information
    assistant = Assistants.create(
        model='qwen-max',  # 此处以qwen-max为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        name='smart helper',
        description='A tool helper.',
        instructions='You are a helpful assistant.',  # noqa E501
    )

    return assistant


def verify_status_code(res):
    if res.status_code != HTTPStatus.OK:
        print('Failed: ')
        print(res)
        sys.exit(res.status_code)


if __name__ == '__main__':
    # create assistant
    assistant = create_assistant()
    print(assistant)
    verify_status_code(assistant)

    # create thread.
    thread = Threads.create(
        messages=[{
            'role': 'user',
            'content': '如何做出美味的牛肉炖土豆？'
        }])
    print(thread)
    verify_status_code(thread)

    # create run
    run = Runs.create(thread.id, assistant_id=assistant.id)
    print(run)
    verify_status_code(run)
    # wait for run completed or requires_action
    run_status = Runs.wait(run.id, thread_id=thread.id)
    print(run_status)
 
    # get the thread messages.
    msgs = Messages.list(thread.id)
    print(msgs)
    print(json.dumps(msgs, ensure_ascii=False, default=lambda o: o.__dict__, sort_keys=True, indent=4))
