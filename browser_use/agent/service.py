from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
	BaseMessage,
	HumanMessage,
	SystemMessage,
)

# from lmnr.sdk.decorators import observe
from pydantic import BaseModel, ValidationError

from browser_use.agent.gif import create_history_gif
from browser_use.agent.message_manager.service import MessageManager, MessageManagerSettings
from browser_use.agent.message_manager.utils import convert_input_messages, extract_json_from_model_output, save_conversation
from browser_use.agent.prompts import AgentMessagePrompt, PlannerPrompt, SystemPrompt
from browser_use.agent.views import (
	ActionResult,
	AgentError,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
	AgentSettings,
	AgentState,
	AgentStepInfo,
	StepMetadata,
	ToolCallingMethod,
)
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserState, BrowserStateHistory
from browser_use.controller.registry.views import ActionModel
from browser_use.controller.service import Controller
from browser_use.dom.history_tree_processor.service import (
	DOMHistoryElement,
	HistoryTreeProcessor,
)
from browser_use.telemetry.service import ProductTelemetry
from browser_use.telemetry.views import (
	AgentEndTelemetryEvent,
	AgentRunTelemetryEvent,
	AgentStepTelemetryEvent,
)
from browser_use.utils import time_execution_async, time_execution_sync

load_dotenv()
logger = logging.getLogger(__name__)


def log_response(response: AgentOutput) -> None:
	"""Utility function to log the model's response."""

	if 'Success' in response.current_state.evaluation_previous_goal:
		emoji = '👍'
	elif 'Failed' in response.current_state.evaluation_previous_goal:
		emoji = '⚠'
	else:
		emoji = '🤷'

	logger.info(f'{emoji} Eval: {response.current_state.evaluation_previous_goal}')
	logger.info(f'🧠 Memory: {response.current_state.memory}')
	logger.info(f'🎯 Next goal: {response.current_state.next_goal}')
	for i, action in enumerate(response.action):
		logger.info(f'🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}')


Context = TypeVar('Context')


class Agent(Generic[Context]):
	@time_execution_sync('--init (agent)')
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		# Optional parameters
		browser: Browser | None = None,
		browser_context: BrowserContext | None = None,
		controller: Controller[Context] = Controller(),
		# Initial agent run parameters
		sensitive_data: Optional[Dict[str, str]] = None,
		initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
		# Cloud Callbacks
		register_new_step_callback: Callable[['BrowserState', 'AgentOutput', int], Awaitable[None]] | None = None,
		register_done_callback: Callable[['AgentHistoryList'], Awaitable[None]] | None = None,
		register_external_agent_status_raise_error_callback: Callable[[], Awaitable[bool]] | None = None,
		# Agent settings
		use_vision: bool = True,
		use_vision_for_planner: bool = False,
		save_conversation_path: Optional[str] = None,
		save_conversation_path_encoding: Optional[str] = 'utf-8',
		max_failures: int = 3,
		retry_delay: int = 10,
		override_system_message: Optional[str] = None,
		extend_system_message: Optional[str] = None,
		max_input_tokens: int = 128000,
		validate_output: bool = False,
		message_context: Optional[str] = None,
		generate_gif: bool | str = False,
		available_file_paths: Optional[list[str]] = None,
		include_attributes: list[str] = [
			'title',
			'type',
			'name',
			'role',
			'aria-label',
			'placeholder',
			'value',
			'alt',
			'aria-expanded',
			'data-date-format',
		],
		max_actions_per_step: int = 10,
		tool_calling_method: Optional[ToolCallingMethod] = 'auto',
		page_extraction_llm: Optional[BaseChatModel] = None,
		planner_llm: Optional[BaseChatModel] = None,
		planner_interval: int = 1,  # Run planner every N steps
		# Inject state
		injected_agent_state: Optional[AgentState] = None,
		#
		context: Context | None = None,
	):
		if page_extraction_llm is None:
			page_extraction_llm = llm

		# Core components
		self.task = task
		self.llm = llm
		self.controller = controller
		self.sensitive_data = sensitive_data

		self.settings = AgentSettings(
			use_vision=use_vision,
			use_vision_for_planner=use_vision_for_planner,
			save_conversation_path=save_conversation_path,
			save_conversation_path_encoding=save_conversation_path_encoding,
			max_failures=max_failures,
			retry_delay=retry_delay,
			override_system_message=override_system_message,
			extend_system_message=extend_system_message,
			max_input_tokens=max_input_tokens,
			validate_output=validate_output,
			message_context=message_context,
			generate_gif=generate_gif,
			available_file_paths=available_file_paths,
			include_attributes=include_attributes,
			max_actions_per_step=max_actions_per_step,
			tool_calling_method=tool_calling_method,
			page_extraction_llm=page_extraction_llm,
			planner_llm=planner_llm,
			planner_interval=planner_interval,
		)

		# Initialize state
		self.state = injected_agent_state or AgentState()

		# Action setup
		self._setup_action_models()
		self._set_browser_use_version_and_source()
		self.initial_actions = self._convert_initial_actions(initial_actions) if initial_actions else None

		# Model setup
		self._set_model_names()

		# for models without tool calling, add available actions to context
		self.available_actions = self.controller.registry.get_prompt_description()

		self.tool_calling_method = self._set_tool_calling_method()
		self.settings.message_context = self._set_message_context()

		# Initialize message manager with state
		self._message_manager = MessageManager(
			task=task,
			system_message=SystemPrompt(
				action_description=self.available_actions,
				max_actions_per_step=self.settings.max_actions_per_step,
				override_system_message=override_system_message,
				extend_system_message=extend_system_message,
			).get_system_message(),
			settings=MessageManagerSettings(
				max_input_tokens=self.settings.max_input_tokens,
				include_attributes=self.settings.include_attributes,
				message_context=self.settings.message_context,
				sensitive_data=sensitive_data,
				available_file_paths=self.settings.available_file_paths,
			),
			state=self.state.message_manager_state,
		)

		# Browser setup
		self.injected_browser = browser is not None
		self.injected_browser_context = browser_context is not None
		self.browser = browser if browser is not None else (None if browser_context else Browser())
		if browser_context:
			self.browser_context = browser_context
		elif self.browser:
			self.browser_context = BrowserContext(browser=self.browser, config=self.browser.config.new_context_config)
		else:
			self.browser = Browser()
			self.browser_context = BrowserContext(browser=self.browser)

		# Callbacks
		self.register_new_step_callback = register_new_step_callback
		self.register_done_callback = register_done_callback
		self.register_external_agent_status_raise_error_callback = register_external_agent_status_raise_error_callback

		# Context
		self.context = context

		# Telemetry
		self.telemetry = ProductTelemetry()

		if self.settings.save_conversation_path:
			logger.info(f'Saving conversation to {self.settings.save_conversation_path}')

	def _set_message_context(self) -> str | None:
		if self.tool_calling_method == 'raw':
			if self.settings.message_context:
				self.settings.message_context += f'\n\nAvailable actions: {self.available_actions}'
			else:
				self.settings.message_context = f'Available actions: {self.available_actions}'
		return self.settings.message_context

	def _set_browser_use_version_and_source(self) -> None:
		"""Get the version and source of the browser-use package (git or pip in a nutshell)"""
		try:
			# First check for repository-specific files
			repo_files = ['.git', 'README.md', 'docs', 'examples']
			package_root = Path(__file__).parent.parent.parent

			# If all of these files/dirs exist, it's likely from git
			if all(Path(package_root / file).exists() for file in repo_files):
				try:
					import subprocess

					version = subprocess.check_output(['git', 'describe', '--tags']).decode('utf-8').strip()
				except Exception:
					version = 'unknown'
				source = 'git'
			else:
				# If no repo files found, try getting version from pip
				import pkg_resources

				version = pkg_resources.get_distribution('browser-use').version
				source = 'pip'
		except Exception:
			version = 'unknown'
			source = 'unknown'

		logger.debug(f'Version: {version}, Source: {source}')
		self.version = version
		self.source = source

	def _set_model_names(self) -> None:
		self.chat_model_library = self.llm.__class__.__name__
		self.model_name = 'Unknown'
		if hasattr(self.llm, 'model_name'):
			model = self.llm.model_name  # type: ignore
			self.model_name = model if model is not None else 'Unknown'
		elif hasattr(self.llm, 'model'):
			model = self.llm.model  # type: ignore
			self.model_name = model if model is not None else 'Unknown'

		if self.settings.planner_llm:
			if hasattr(self.settings.planner_llm, 'model_name'):
				self.planner_model_name = self.settings.planner_llm.model_name  # type: ignore
			elif hasattr(self.settings.planner_llm, 'model'):
				self.planner_model_name = self.settings.planner_llm.model  # type: ignore
			else:
				self.planner_model_name = 'Unknown'
		else:
			self.planner_model_name = None

	def _setup_action_models(self) -> None:
		"""Setup dynamic action models from controller's registry"""
		self.ActionModel = self.controller.registry.create_action_model()
		# Create output model with the dynamic actions
		self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)

		# used to force the done action when max_steps is reached
		self.DoneActionModel = self.controller.registry.create_action_model(include_actions=['done'])
		self.DoneAgentOutput = AgentOutput.type_with_custom_actions(self.DoneActionModel)

	def _set_tool_calling_method(self) -> Optional[ToolCallingMethod]:
		tool_calling_method = self.settings.tool_calling_method
		if tool_calling_method == 'auto':
			if self.model_name == 'deepseek-reasoner' or self.model_name.startswith('deepseek-r1'):
				return 'raw'
			elif self.chat_model_library == 'ChatGoogleGenerativeAI':
				return None
			elif self.chat_model_library == 'ChatOpenAI':
				return 'function_calling'
			elif self.chat_model_library == 'AzureChatOpenAI':
				return 'function_calling'
			else:
				return None
		else:
			return tool_calling_method

	def add_new_task(self, new_task: str) -> None:
		self._message_manager.add_new_task(new_task)

	async def _raise_if_stopped_or_paused(self) -> None:
		"""Utility function that raises an InterruptedError if the agent is stopped or paused."""
		#如果代理停止或暂停，则引发InterruptedError的实用程序函数
		if self.register_external_agent_status_raise_error_callback:
			if await self.register_external_agent_status_raise_error_callback():
				raise InterruptedError

		if self.state.stopped or self.state.paused:
			logger.debug('Agent paused after getting state')
			raise InterruptedError

	# @observe(name='agent.step', ignore_output=True, ignore_input=True)
	@time_execution_async('--step (agent)')
	async def step(self, step_info: Optional[AgentStepInfo] = None) -> None:
		"""Execute one step of the task"""
		#完成一步任务

		#初始化几个变量
		logger.info(f'📍 Step {self.state.n_steps}')
		state = None
		model_output = None
		result: list[ActionResult] = []
		step_start_time = time.time()
		tokens = 0

		try:
			state = await self.browser_context.get_state()#获取浏览器的当前状态

			await self._raise_if_stopped_or_paused()#检查代理是否应该停止或暂停

			self._message_manager.add_state_message(state, self.state.last_result, step_info, self.settings.use_vision)
			#将当前状态、上一步的结果和步骤信息添加到消息管理器中，消息管理器维护代理与模型之间的对话

			# Run planner at specified intervals if planner is configured
			#如果配置了规划器，则以指定的时间间隔运行规划器(没太看懂这个时间间隔运行规划器是干嘛的)
			if self.settings.planner_llm and self.state.n_steps % self.settings.planner_interval == 0:
				plan = await self._run_planner()
				# add plan before last state message
				#在最后一条状态消息之前添加计划
				self._message_manager.add_plan(plan, position=-1)

			if step_info and step_info.is_last_step():
				# Add last step warning if needed
				msg = 'Now comes your last step. Use only the "done" action now. No other actions - so here your action sequence musst have length 1.'
				msg += '\nIf the task is not yet fully finished as requested by the user, set success in "done" to false! E.g. if not all steps are fully completed.'
				msg += '\nIf the task is fully finished, set success in "done" to true.'
				msg += '\nInclude everything you found out for the ultimate task in the done text.'
				logger.info('Last step finishing up')
				self._message_manager._add_message_with_tokens(HumanMessage(content=msg))
				self.AgentOutput = self.DoneAgentOutput
			#如果这是最后一步，添加特殊指令告诉模型这是最后一步，要求它只使用"done"动作，并根据任务完成情况设置成功或失败标志。同时切换输出类型为 DoneAgentOutput

			input_messages = self._message_manager.get_messages()
			tokens = self._message_manager.state.history.current_tokens

			try:
				model_output = await self.get_next_action(input_messages)
				#调用模型，获取下一步要执行的动作
				self.state.n_steps += 1

				if self.register_new_step_callback:
					await self.register_new_step_callback(state, model_output, self.state.n_steps)
				#如果注册了步骤回调函数，调用它并传递当前状态、模型输出和步骤编号

				if self.settings.save_conversation_path:
					target = self.settings.save_conversation_path + f'_{self.state.n_steps}.txt'
					save_conversation(input_messages, model_output, target, self.settings.save_conversation_path_encoding)
				#如果配置了保存对话，将当前对话保存到文件中

				self._message_manager._remove_last_state_message()  # we dont want the whole state in the chat history
				#从消息历史中移除最后的状态消息（可能太大），避免累积过多的状态信息

				await self._raise_if_stopped_or_paused()

				self._message_manager.add_model_output(model_output)
			except Exception as e:
				# model call failed, remove last state message from history
				#如果模型调用出错，移除最后的状态消息，然后重新抛出异常
				self._message_manager._remove_last_state_message()
				raise e

			result: list[ActionResult] = await self.multi_act(model_output.action)
		#执行模型输出中指定的动作，使用 multi_act 方法
			self.state.last_result = result

			if len(result) > 0 and result[-1].is_done:
				logger.info(f'📄 Result: {result[-1].extracted_content}')
			#如果最后一个结果标记为已完成，记录其内容，使用文档emoji（📄）增加可视性。

			self.state.consecutive_failures = 0
			#重置连续失败计数，因为这一步成功了
		except InterruptedError:
			logger.debug('Agent paused')
			self.state.last_result = [
				ActionResult(
					error='The agent was paused - now continuing actions might need to be repeated', include_in_memory=True
				)
			]
			return
		#处理中断异常，记录代理被暂停的信息，设置带有错误消息的结果，并提前返回
		except Exception as e:
			result = await self._handle_step_error(e)
			self.state.last_result = result
		#处理其他异常，调用错误处理方法，并保存错误结果。

		finally:
			step_end_time = time.time()
			actions = [a.model_dump(exclude_unset=True) for a in model_output.action] if model_output else []
			self.telemetry.capture(
				AgentStepTelemetryEvent(
					agent_id=self.state.agent_id,
					step=self.state.n_steps,
					actions=actions,
					consecutive_failures=self.state.consecutive_failures,
					step_error=[r.error for r in result if r.error] if result else ['No result'],
				)
			)
			if not result:
				return

			if state:
				metadata = StepMetadata(
					step_number=self.state.n_steps,
					step_start_time=step_start_time,
					step_end_time=step_end_time,
					input_tokens=tokens,
				)
				self._make_history_item(model_output, state, result, metadata)

	@time_execution_async('--handle_step_error (agent)')
	async def _handle_step_error(self, error: Exception) -> list[ActionResult]:
		"""Handle all types of errors that can occur during a step"""
		include_trace = logger.isEnabledFor(logging.DEBUG)
		error_msg = AgentError.format_error(error, include_trace=include_trace)
		prefix = f'❌ Result failed {self.state.consecutive_failures + 1}/{self.settings.max_failures} times:\n '

		if isinstance(error, (ValidationError, ValueError)):
			logger.error(f'{prefix}{error_msg}')
			if 'Max token limit reached' in error_msg:
				# cut tokens from history
				self._message_manager.settings.max_input_tokens = self.settings.max_input_tokens - 500
				logger.info(
					f'Cutting tokens from history - new max input tokens: {self._message_manager.settings.max_input_tokens}'
				)
				self._message_manager.cut_messages()
			elif 'Could not parse response' in error_msg:
				# give model a hint how output should look like
				error_msg += '\n\nReturn a valid JSON object with the required fields.'

			self.state.consecutive_failures += 1
		else:
			from google.api_core.exceptions import ResourceExhausted
			from openai import RateLimitError

			if isinstance(error, RateLimitError) or isinstance(error, ResourceExhausted):
				logger.warning(f'{prefix}{error_msg}')
				await asyncio.sleep(self.settings.retry_delay)
				self.state.consecutive_failures += 1
			else:
				logger.error(f'{prefix}{error_msg}')
				self.state.consecutive_failures += 1

		return [ActionResult(error=error_msg, include_in_memory=True)]

	def _make_history_item(
		self,
		model_output: AgentOutput | None,
		state: BrowserState,
		result: list[ActionResult],
		metadata: Optional[StepMetadata] = None,
	) -> None:
		"""Create and store history item"""

		if model_output:
			interacted_elements = AgentHistory.get_interacted_element(model_output, state.selector_map)
		else:
			interacted_elements = [None]

		state_history = BrowserStateHistory(
			url=state.url,
			title=state.title,
			tabs=state.tabs,
			interacted_element=interacted_elements,
			screenshot=state.screenshot,
		)

		history_item = AgentHistory(model_output=model_output, result=result, state=state_history, metadata=metadata)

		self.state.history.history.append(history_item)

	THINK_TAGS = re.compile(r'<think>.*?</think>', re.DOTALL)

	def _remove_think_tags(self, text: str) -> str:
		"""Remove think tags from text"""
		return re.sub(self.THINK_TAGS, '', text)

	def _convert_input_messages(self, input_messages: list[BaseMessage]) -> list[BaseMessage]:
		"""Convert input messages to the correct format"""
		if self.model_name == 'deepseek-reasoner' or self.model_name.startswith('deepseek-r1'):
			return convert_input_messages(input_messages, self.model_name)
		else:
			return input_messages

	@time_execution_async('--get_next_action (agent)')
	async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
		"""Get next action from LLM based on current state"""
		input_messages = self._convert_input_messages(input_messages)

		if self.tool_calling_method == 'raw':
			output = self.llm.invoke(input_messages)
			# TODO: currently invoke does not return reasoning_content, we should override invoke
			output.content = self._remove_think_tags(str(output.content))
			try:
				parsed_json = extract_json_from_model_output(output.content)
				parsed = self.AgentOutput(**parsed_json)
			except (ValueError, ValidationError) as e:
				logger.warning(f'Failed to parse model output: {output} {str(e)}')
				raise ValueError('Could not parse response.')

		elif self.tool_calling_method is None:
			structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True)
			response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
			parsed: AgentOutput | None = response['parsed']
		else:
			structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True, method=self.tool_calling_method)
			response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
			parsed: AgentOutput | None = response['parsed']

		if parsed is None:
			raise ValueError('Could not parse response.')

		# cut the number of actions to max_actions_per_step if needed
		if len(parsed.action) > self.settings.max_actions_per_step:
			parsed.action = parsed.action[: self.settings.max_actions_per_step]

		log_response(parsed)

		return parsed

	def _log_agent_run(self) -> None:
		"""Log the agent run"""
		#记录代理运行的开始，可能包括任务信息、时间戳等
		logger.info(f'🚀 Starting task: {self.task}')

		logger.debug(f'Version: {self.version}, Source: {self.source}')
		self.telemetry.capture(
			AgentRunTelemetryEvent(
				agent_id=self.state.agent_id,
				use_vision=self.settings.use_vision,
				task=self.task,
				model_name=self.model_name,
				chat_model_library=self.chat_model_library,
				version=self.version,
				source=self.source,
			)
		)

	async def take_step(self) -> tuple[bool, bool]:
		"""Take a step

		Returns:
			Tuple[bool, bool]: (is_done, is_valid)
		"""
		await self.step()

		if self.state.history.is_done():
			if self.settings.validate_output:
				if not await self._validate_output():
					return True, False

			await self.log_completion()
			if self.register_done_callback:
				await self.register_done_callback(self.state.history)

			return True, True

		return False, False

	# @observe(name='agent.run', ignore_output=True)
	@time_execution_async('--run (agent)')
	async def run(self, max_steps: int = 100) -> AgentHistoryList:
		"""Execute the task with maximum number of steps"""
		#定义一个异步方法 run，接受一个可选参数 max_steps（默认为100），表示最大执行步骤数。返回类型是 AgentHistoryList，用于记录代理执行的历史
		try:#开始一个 try-finally 块，确保无论执行过程中是否出现异常，都能执行后面finally代码里的必要的清理操作
			self._log_agent_run()
			#记录代理运行的开始，可能包括任务信息、时间戳等
			# Execute initial actions if provided；执行提供的动作
			if self.initial_actions:
				result = await self.multi_act(self.initial_actions, check_for_new_elements=False)
				self.state.last_result = result
			#如果有预设的初始操作（initial_actions），先执行这些操作。调用之前我们讨论过的 multi_act 方法，并禁用新元素检查。将结果存储在代理状态中
			#初始操作不涉及元素检查

			for step in range(max_steps):
				# Check if we should stop due to too many failures
				if self.state.consecutive_failures >= self.settings.max_failures:
					logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
					break
				#检查连续失败次数，如果超过设定阈值，记录错误并中断执行。

				# Check control flags before each step； 在每个步骤之前检查控制标志
				if self.state.stopped:#这个参数在下面的stop函数里，由日志信息决定他的ture&false
					logger.info('Agent stopped')
					break

				while self.state.paused:
					await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
					if self.state.stopped:  # Allow stopping while paused
						break
				#如果代理被暂停，进入等待循环，每0.2秒检查一次是否恢复或停止。这样设计可以响应外部控制命令，同时避免CPU资源浪费

				step_info = AgentStepInfo(step_number=step, max_steps=max_steps)
				await self.step(step_info)
				#步骤信息对象，包含当前步骤号和最大步骤数，然后执行单个步骤

				if self.state.history.is_done():
					if self.settings.validate_output and step < max_steps - 1:
					#如果启用了输出验证且还有步骤可用，验证输出结果。如果验证失败，继续下一个步骤
						if not await self._validate_output():
							continue

					await self.log_completion()
					break
			else:
				logger.info('❌ Failed to complete task in maximum steps')

			return self.state.history
		finally:
			self.telemetry.capture(
				#捕获并记录代理结束的遥测事件，包含多项执行统计信息，如执行是否成功、步骤数、错误信息、token 消耗和总执行时间等
				AgentEndTelemetryEvent(
					agent_id=self.state.agent_id,
					is_done=self.state.history.is_done(),
					success=self.state.history.is_successful(),
					steps=self.state.n_steps,
					max_steps_reached=self.state.n_steps >= max_steps,
					errors=self.state.history.errors(),
					total_input_tokens=self.state.history.total_input_tokens(),
					total_duration_seconds=self.state.history.total_duration_seconds(),
				)
			)

			if not self.injected_browser_context:#如果浏览器上下文不是从外部注入的，关闭它以释放资源
				await self.browser_context.close()

			if not self.injected_browser and self.browser:#如果浏览器实例不是从外部注入的且存在，关闭它以释放资源
				await self.browser.close()

			if self.settings.generate_gif:
				output_path: str = 'agent_history.gif'
				if isinstance(self.settings.generate_gif, str):
					output_path = self.settings.generate_gif
			#如果启用了 GIF 生成功能，创建一个可视化的执行历史 GIF 文件。根据设置，使用默认路径或指定的路径。
				create_history_gif(task=self.task, history=self.state.history, output_path=output_path)

	# @observe(name='controller.multi_act')
	@time_execution_async('--multi-act (agent)')
	async def multi_act(
		self,
		actions: list[ActionModel],
		check_for_new_elements: bool = True,
	) -> list[ActionResult]:
		"""Execute multiple actions"""
		results = []#初始化一个空列表来存储每个操作的结果

		cached_selector_map = await self.browser_context.get_selector_map() 
		#异步获取当前页面的元素选择器映射（selector map），这是一个当前页面元素的索引
		cached_path_hashes = set(e.hash.branch_path_hash for e in cached_selector_map.values())
		#从选择器映射中提取所有元素的分支路径哈希值
		await self.browser_context.remove_highlights()
		#删除由HighlightElement函数创建的所有高亮叠加和标签。处理页面可能关闭或无法访问的情况

		for i, action in enumerate(actions):#遍历操作列表
			if action.get_index() is not None and i != 0:
				new_state = await self.browser_context.get_state()#如果不是首个动作即不是新的标签页，就获取当前的状态（因为不是首个动作，说明执行了一个动作，需要更新状态
				new_path_hashes = set(e.hash.branch_path_hash for e in new_state.selector_map.values())
				if check_for_new_elements and not new_path_hashes.issubset(cached_path_hashes):
					#如果页面有新元素出现
					# next action requires index but there are new elements on the page
					#记录日志，添加一个结果说明页面出现了变化，并中断操作循环。这是一种安全机制，防止在页面变化后执行可能不再有效的操作
					msg = f'Something new appeared after action {i} / {len(actions)}'
					logger.info(msg)
					results.append(ActionResult(extracted_content=msg, include_in_memory=True))
					break
			
			await self._raise_if_stopped_or_paused()
			#检查操作是否应该停止或暂停，是的话执行InterruptedError（os系统操作函数）相当于用户的ctrl+c行为
			result = await self.controller.act(#调用控制器的 act 方法，传递多个参数，包括当前操作、浏览器上下文、语言模型、敏感数据、可用文件路径和上下文
				action,
				self.browser_context,
				self.settings.page_extraction_llm,
				self.sensitive_data,
				self.settings.available_file_paths,
				context=self.context,
			)

			results.append(result)#将操作结果添加到开始时创建的结果列表中

			logger.debug(f'Executed action {i + 1} / {len(actions)}')
			if results[-1].is_done or results[-1].error or i == len(actions) - 1:
				break
				#如果最后一个操作结果标记为已完成、发生错误，或者这是最后一个操作，则中断循环

			await asyncio.sleep(self.browser_context.config.wait_between_actions)
			# hash all elements. if it is a subset of cached_state its fine - else break (new elements on page)
			#如果新元素集是旧集合的子集，继续执行；否则停止，因为页面有变化。
		return results

	async def _validate_output(self) -> bool:
		"""Validate the output of the last action is what the user wanted"""
		system_msg = (
			f'You are a validator of an agent who interacts with a browser. '
			f'Validate if the output of last action is what the user wanted and if the task is completed. '
			f'If the task is unclear defined, you can let it pass. But if something is missing or the image does not show what was requested dont let it pass. '
			f'Try to understand the page and help the model with suggestions like scroll, do x, ... to get the solution right. '
			f'Task to validate: {self.task}. Return a JSON object with 2 keys: is_valid and reason. '
			f'is_valid is a boolean that indicates if the output is correct. '
			f'reason is a string that explains why it is valid or not.'
			f' example: {{"is_valid": false, "reason": "The user wanted to search for "cat photos", but the agent searched for "dog photos" instead."}}'
		)

		if self.browser_context.session:
			state = await self.browser_context.get_state()
			content = AgentMessagePrompt(
				state=state,
				result=self.state.last_result,
				include_attributes=self.settings.include_attributes,
			)
			msg = [SystemMessage(content=system_msg), content.get_user_message(self.settings.use_vision)]
		else:
			# if no browser session, we can't validate the output
			return True

		class ValidationResult(BaseModel):
			"""
			Validation results.
			"""

			is_valid: bool
			reason: str

		validator = self.llm.with_structured_output(ValidationResult, include_raw=True)
		response: dict[str, Any] = await validator.ainvoke(msg)  # type: ignore
		parsed: ValidationResult = response['parsed']
		is_valid = parsed.is_valid
		if not is_valid:
			logger.info(f'❌ Validator decision: {parsed.reason}')
			msg = f'The output is not yet correct. {parsed.reason}.'
			self.state.last_result = [ActionResult(extracted_content=msg, include_in_memory=True)]
		else:
			logger.info(f'✅ Validator decision: {parsed.reason}')
		return is_valid

	async def log_completion(self) -> None:
		"""Log the completion of the task"""
		logger.info('✅ Task completed')
		if self.state.history.is_successful():
			logger.info('✅ Successfully')
		else:
			logger.info('❌ Unfinished')

		if self.register_done_callback:
			await self.register_done_callback(self.state.history)

	async def rerun_history(
		self,
		history: AgentHistoryList,
		max_retries: int = 3,
		skip_failures: bool = True,
		delay_between_actions: float = 2.0,
	) -> list[ActionResult]:
		"""
		Rerun a saved history of actions with error handling and retry logic.

		Args:
				history: The history to replay
				max_retries: Maximum number of retries per action
				skip_failures: Whether to skip failed actions or stop execution
				delay_between_actions: Delay between actions in seconds

		Returns:
				List of action results
		"""
		# Execute initial actions if provided
		if self.initial_actions:
			result = await self.multi_act(self.initial_actions)
			self.state.last_result = result

		results = []

		for i, history_item in enumerate(history.history):
			goal = history_item.model_output.current_state.next_goal if history_item.model_output else ''
			logger.info(f'Replaying step {i + 1}/{len(history.history)}: goal: {goal}')

			if (
				not history_item.model_output
				or not history_item.model_output.action
				or history_item.model_output.action == [None]
			):
				logger.warning(f'Step {i + 1}: No action to replay, skipping')
				results.append(ActionResult(error='No action to replay'))
				continue

			retry_count = 0
			while retry_count < max_retries:
				try:
					result = await self._execute_history_step(history_item, delay_between_actions)
					results.extend(result)
					break

				except Exception as e:
					retry_count += 1
					if retry_count == max_retries:
						error_msg = f'Step {i + 1} failed after {max_retries} attempts: {str(e)}'
						logger.error(error_msg)
						if not skip_failures:
							results.append(ActionResult(error=error_msg))
							raise RuntimeError(error_msg)
					else:
						logger.warning(f'Step {i + 1} failed (attempt {retry_count}/{max_retries}), retrying...')
						await asyncio.sleep(delay_between_actions)

		return results

	async def _execute_history_step(self, history_item: AgentHistory, delay: float) -> list[ActionResult]:
		"""Execute a single step from history with element validation"""
		state = await self.browser_context.get_state()
		if not state or not history_item.model_output:
			raise ValueError('Invalid state or model output')
		updated_actions = []
		for i, action in enumerate(history_item.model_output.action):
			updated_action = await self._update_action_indices(
				history_item.state.interacted_element[i],
				action,
				state,
			)
			updated_actions.append(updated_action)

			if updated_action is None:
				raise ValueError(f'Could not find matching element {i} in current page')

		result = await self.multi_act(updated_actions)

		await asyncio.sleep(delay)
		return result

	async def _update_action_indices(
		self,
		historical_element: Optional[DOMHistoryElement],
		action: ActionModel,  # Type this properly based on your action model
		current_state: BrowserState,
	) -> Optional[ActionModel]:
		"""
		Update action indices based on current page state.
		Returns updated action or None if element cannot be found.
		"""
		if not historical_element or not current_state.element_tree:
			return action

		current_element = HistoryTreeProcessor.find_history_element_in_tree(historical_element, current_state.element_tree)

		if not current_element or current_element.highlight_index is None:
			return None

		old_index = action.get_index()
		if old_index != current_element.highlight_index:
			action.set_index(current_element.highlight_index)
			logger.info(f'Element moved in DOM, updated index from {old_index} to {current_element.highlight_index}')

		return action

	async def load_and_rerun(self, history_file: Optional[str | Path] = None, **kwargs) -> list[ActionResult]:
		"""
		Load history from file and rerun it.

		Args:
				history_file: Path to the history file
				**kwargs: Additional arguments passed to rerun_history
		"""
		if not history_file:
			history_file = 'AgentHistory.json'
		history = AgentHistoryList.load_from_file(history_file, self.AgentOutput)
		return await self.rerun_history(history, **kwargs)

	def save_history(self, file_path: Optional[str | Path] = None) -> None:
		"""Save the history to a file"""
		if not file_path:
			file_path = 'AgentHistory.json'
		self.state.history.save_to_file(file_path)

	def pause(self) -> None:
		"""Pause the agent before the next step"""
		logger.info('🔄 pausing Agent ')
		self.state.paused = True

	def resume(self) -> None:
		"""Resume the agent"""
		logger.info('▶️ Agent resuming')
		self.state.paused = False

	def stop(self) -> None:
		"""Stop the agent"""
		logger.info('⏹️ Agent stopping')
		self.state.stopped = True

	def _convert_initial_actions(self, actions: List[Dict[str, Dict[str, Any]]]) -> List[ActionModel]:
		"""Convert dictionary-based actions to ActionModel instances"""
		converted_actions = []
		action_model = self.ActionModel
		for action_dict in actions:
			# Each action_dict should have a single key-value pair
			action_name = next(iter(action_dict))
			params = action_dict[action_name]

			# Get the parameter model for this action from registry
			action_info = self.controller.registry.registry.actions[action_name]
			param_model = action_info.param_model

			# Create validated parameters using the appropriate param model
			validated_params = param_model(**params)

			# Create ActionModel instance with the validated parameters
			action_model = self.ActionModel(**{action_name: validated_params})
			converted_actions.append(action_model)

		return converted_actions

	async def _run_planner(self) -> Optional[str]:
		"""Run the planner to analyze state and suggest next steps"""
		#运行计划程序以分析状态并建议下一步
		# Skip planning if no planner_llm is set
		if not self.settings.planner_llm:
			return None

		# Create planner message history using full message history
		#使用完整的消息历史记录创建计划者消息历史记录
		planner_messages = [
			PlannerPrompt(self.controller.registry.get_prompt_description()).get_system_message(),
			*self._message_manager.get_messages()[1:],  
			#首先添加一个系统消息，使用 PlannerPrompt 类和控制器的注册表中的提示描述（就是控制器里的提示词描述）
			# Use full message history except the first
			#然后添加完整的消息历史（除了第一个消息，通常是系统消息）
		]

		if not self.settings.use_vision_for_planner and self.settings.use_vision:
			last_state_message: HumanMessage = planner_messages[-1]
			# remove image from last state message
			#如果代理设置了使用视觉功能，但计划器不应使用视觉功能，那么需要处理最后一个状态消息（即去掉图片信息，保留文字
			new_msg = ''
			if isinstance(last_state_message.content, list):
				for msg in last_state_message.content:
					if msg['type'] == 'text':  # type: ignore
						new_msg += msg['text']  # type: ignore
					elif msg['type'] == 'image_url':  # type: ignore
						continue  # type: ignore
			else:
				new_msg = last_state_message.content

			planner_messages[-1] = HumanMessage(content=new_msg)

		planner_messages = convert_input_messages(planner_messages, self.planner_model_name)

		# Get planner output
		response = await self.settings.planner_llm.ainvoke(planner_messages)
		plan = str(response.content)
		# if deepseek-reasoner, remove think tags
		#如果使用的是 "deepseek-reasoner" 模型，调用一个辅助方法 _remove_think_tags 移除响应中的思考标签。这些标签可能是模型输出的一部分，但不需要包含在最终计划中。
		#去掉推理模型中的think内容
		if self.planner_model_name == 'deepseek-reasoner':
			plan = self._remove_think_tags(plan)
		try:#尝试将计划解析为 JSON 对象，如果成功，以格式化的方式记录计划内容。
			plan_json = json.loads(plan)
			logger.info(f'Planning Analysis:\n{json.dumps(plan_json, indent=4)}')
		except json.JSONDecodeError:#如果计划不是有效的 JSON 格式，直接记录原始计划文本。
			logger.info(f'Planning Analysis:\n{plan}')
		except Exception as e:#处理其他可能的异常，记录错误信息和原始计划。
			logger.debug(f'Error parsing planning analysis: {e}')
			logger.info(f'Plan: {plan}')

		return plan

	@property
	def message_manager(self) -> MessageManager:
		return self._message_manager
