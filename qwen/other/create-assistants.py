from dashscope import Assistants
import os
import dashscope
from dotenv import load_dotenv

load_dotenv()
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
assistant = Assistants.create(
        # 建议您优先配置环境变量。若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 此处以qwen-max为例，可按需更换模型名称。模型列表：https://www.alibabacloud.com/help/zh/model-studio/getting-started/models
        model='qwen-max',
        name='smart helper',
        description='A tool helper.',
        instructions='You are a helpful assistant. When asked a question, use tools wherever possible.', 
        tools=[{
            'type': 'search'
        }, {
            'type': 'function',
            'function': {
                'name': 'big_add',
                'description': 'Add to number',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'left': {
                            'type': 'integer',
                            'description': 'The left operator'
                        },
                        'right': {
                            'type': 'integer',
                            'description': 'The right operator.'
                        }
                    },
                    'required': ['left', 'right']
                }
            }
        }],
)
print(assistant)