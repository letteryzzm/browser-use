import asyncio
from inspect import iscoroutinefunction, signature
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, create_model

from browser_use.browser.context import BrowserContext
from browser_use.controller.registry.views import (
	ActionModel,
	ActionRegistry,
	RegisteredAction,
)
from browser_use.telemetry.service import ProductTelemetry
from browser_use.telemetry.views import (
	ControllerRegisteredFunctionsTelemetryEvent,
	RegisteredFunction,
)
from browser_use.utils import time_execution_async, time_execution_sync

Context = TypeVar('Context')#定义一个类型变量Context，用于泛型编程


class Registry(Generic[Context]):
	"""Service for registering and managing actions"""
		#定义Registry类，它是泛型的，可以使用不同类型的上下文
	def __init__(self, exclude_actions: list[str] = []):
		self.registry = ActionRegistry()
		self.telemetry = ProductTelemetry()
		self.exclude_actions = exclude_actions

	@time_execution_sync('--create_param_model')
	def _create_param_model(self, function: Callable) -> Type[BaseModel]:
		"""Creates a Pydantic model from function signature"""#从函数签名自动创建Pydantic模型
		sig = signature(function)
		params = {
			name: (param.annotation, ... if param.default == param.empty else param.default)
			for name, param in sig.parameters.items()
			if name != 'browser' and name != 'page_extraction_llm' and name != 'available_file_paths'
			#排除特定参数（browser、page_extraction_llm、available_file_paths）
		}
		# TODO: make the types here work
		return create_model(#创建并返回Pydantic模型，以ActionModel为基类
			f'{function.__name__}_parameters',
			__base__=ActionModel,
			**params,  # type: ignore
		)

	def action(
		self,
		description: str,
		param_model: Optional[Type[BaseModel]] = None,
	):
		"""Decorator for registering actions"""#定义一个装饰器方法，用于注册动作

		def decorator(func: Callable):
			# Skip registration if action is in exclude_actions 如果函数名在排除列表中，则跳过注册
			if func.__name__ in self.exclude_actions:
				return func

			# Create param model from function if not provided 如果没有提供参数模型，则从函数签名创建
			actual_param_model = param_model or self._create_param_model(func)

			# Wrap sync functions to make them async 检查函数是否为协程函数（异步函数）如果不是，将其包装为异步函数
			if not iscoroutinefunction(func):

				async def async_wrapper(*args, **kwargs):
					return await asyncio.to_thread(func, *args, **kwargs)

				# Copy the signature and other metadata from the original function复制原始函数的签名和元数据
				async_wrapper.__signature__ = signature(func)
				async_wrapper.__name__ = func.__name__
				async_wrapper.__annotations__ = func.__annotations__
				wrapped_func = async_wrapper
			else:
				wrapped_func = func

			action = RegisteredAction(#创建RegisteredAction对象并存储到注册表中
				name=func.__name__,
				description=description,
				function=wrapped_func,
				param_model=actual_param_model,
			)
			self.registry.actions[func.__name__] = action
			return func#返回原始函数，保持装饰器链


		return decorator#返回装饰器函数

	@time_execution_async('--execute_action')
	async def execute_action(#执行注册动作的异步方法
		self,
		action_name: str,
		params: dict,
		browser: Optional[BrowserContext] = None,
		page_extraction_llm: Optional[BaseChatModel] = None,
		sensitive_data: Optional[Dict[str, str]] = None,
		available_file_paths: Optional[list[str]] = None,
		#
		context: Context | None = None,
	) -> Any:
		"""Execute a registered action"""
		if action_name not in self.registry.actions:#检查动作是否存在于注册表中
			raise ValueError(f'Action {action_name} not found')

		action = self.registry.actions[action_name]#获取动作并开始尝试执行
		try:
			# Create the validated Pydantic model
			validated_params = action.param_model(**params)#使用参数模型验证输入参数

			# Check if the first parameter is a Pydantic model
			sig = signature(action.function)#检查函数的第一个参数是否为Pydantic模型
			parameters = list(sig.parameters.values())
			is_pydantic = parameters and issubclass(parameters[0].annotation, BaseModel)
			parameter_names = [param.name for param in parameters]

			if sensitive_data:#如果有敏感数据，替换参数中的敏感数据占位符
				validated_params = self._replace_sensitive_data(validated_params, sensitive_data)

			# Check if the action requires browser 检查动作是否需要浏览器对象
			if 'browser' in parameter_names and not browser:
				raise ValueError(f'Action {action_name} requires browser but none provided.')
			if 'page_extraction_llm' in parameter_names and not page_extraction_llm:
				raise ValueError(f'Action {action_name} requires page_extraction_llm but none provided.')
			if 'available_file_paths' in parameter_names and not available_file_paths:
				raise ValueError(f'Action {action_name} requires available_file_paths but none provided.')

			if 'context' in parameter_names and not context:
				raise ValueError(f'Action {action_name} requires context but none provided.')

			# Prepare arguments based on parameter type准备额外的参数
			extra_args = {}
			if 'context' in parameter_names:
				extra_args['context'] = context
			if 'browser' in parameter_names:
				extra_args['browser'] = browser
			if 'page_extraction_llm' in parameter_names:
				extra_args['page_extraction_llm'] = page_extraction_llm
			if 'available_file_paths' in parameter_names:
				extra_args['available_file_paths'] = available_file_paths
			if action_name == 'input_text' and sensitive_data:
				extra_args['has_sensitive_data'] = True
			if is_pydantic:#根据函数是否接受Pydantic模型决定调用方式 执行动作并返回结果
				return await action.function(validated_params, **extra_args)
			return await action.function(**validated_params.model_dump(), **extra_args)

		except Exception as e:#捕获并重新抛出异常，添加更多上下文信息
			raise RuntimeError(f'Error executing action {action_name}: {str(e)}') from e

	def _replace_sensitive_data(self, params: BaseModel, sensitive_data: Dict[str, str]) -> BaseModel:
		"""Replaces the sensitive data in the params替换参数中的敏感数据"""
		# if there are any str with <secret>placeholder</secret> in the params, replace them with the actual value from sensitive_data

		import re

		secret_pattern = re.compile(r'<secret>(.*?)</secret>')

		def replace_secrets(value): # ...递归替换所有字符串中的敏感数据标记
			if isinstance(value, str):
				matches = secret_pattern.findall(value)
				for placeholder in matches:
					if placeholder in sensitive_data:
						value = value.replace(f'<secret>{placeholder}</secret>', sensitive_data[placeholder])
				return value
			elif isinstance(value, dict):
				return {k: replace_secrets(v) for k, v in value.items()}
			elif isinstance(value, list):
				return [replace_secrets(v) for v in value]
			return value

		for key, value in params.model_dump().items():#处理模型中的每个字段并返回
			params.__dict__[key] = replace_secrets(value)
		return params

	@time_execution_sync('--create_action_model')
	def create_action_model(self, include_actions: Optional[list[str]] = None) -> Type[ActionModel]:
		"""Creates a Pydantic model from registered actions创建包含所有注册动作的Pydantic模型"""
		fields = {#为每个动作创建字段定义
			name: (
				Optional[action.param_model],
				Field(default=None, description=action.description),
			)
			for name, action in self.registry.actions.items()
			if include_actions is None or name in include_actions
		}

		self.telemetry.capture(#记录遥测事件，包含所有注册的函数
			ControllerRegisteredFunctionsTelemetryEvent(
				registered_functions=[
					RegisteredFunction(name=name, params=action.param_model.model_json_schema())
					for name, action in self.registry.actions.items()
					if include_actions is None or name in include_actions
				]
			)
		)

		return create_model('ActionModel', __base__=ActionModel, **fields)  # type:ignore
			#创建并返回包含所有动作的模型
	def get_prompt_description(self) -> str:
		"""Get a description of all actions for the prompt获取所有动作的描述，用于生成提示"""
		return self.registry.get_prompt_description()
