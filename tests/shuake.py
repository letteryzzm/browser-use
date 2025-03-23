import asyncio#导入异步编程库，处理异步操作，允许程序在等待 I/O 完成时执行其他任务。
import os#操作系统接口 os

from dotenv import load_dotenv#用于从 .env 文件加载环境变量
from langchain_openai import ChatOpenAI#LangChain 的 OpenAI 接口，用于与 API 通信
from pydantic import SecretStr#Pydantic 的工具，用于安全地处理敏感信息（如 API 密钥）

from browser_use import Agent
import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)#配置日志系统，设置编码为 UTF-8，日志级别为 INFO
# dotenv
load_dotenv()

api_key = os.getenv('DEEPSEEK_API_KEY', 'sk-522b7a6cf19543efa0e29c075077d4a7')
if not api_key:
	raise ValueError('DEEPSEEK_API_KEY is not set')


async def run_search():#定义一个异步函数 run_search()，用于执行浏览器搜索任务
	agent = Agent(#将下面这些参数输入到Agent文件中去执行
		task=(
			'1. 打开“icourse163.org”这个网站'
			"2. 进入“离散数学” 这门课"
			'3. 找到课件按钮'
			'4. 进行课程观看'
			'5. 把所有课程看完'
		),
		llm=ChatOpenAI(
			base_url='https://api.deepseek.com/v1',
			model='deepseek-chat',
			api_key=SecretStr(api_key),
		),
		use_vision=False,
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(run_search())
