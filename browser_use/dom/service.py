import gc
import json
import logging
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
	from playwright.async_api import Page

from browser_use.dom.views import (
	DOMBaseNode,
	DOMElementNode,
	DOMState,
	DOMTextNode,
	SelectorMap,
)
from browser_use.utils import time_execution_async

logger = logging.getLogger(__name__)


@dataclass
class ViewportInfo:
	width: int
	height: int


class DomService:
	def __init__(self, page: 'Page'):
		self.page = page#定义DomService类，初始化时接收一个Playwright的Page对象
		self.xpath_cache = {}#创建一个XPath缓存字典，可能用于存储已计算的XPath表达式

		self.js_code = resources.read_text('browser_use.dom', 'buildDomTree.js')
		#从资源包中读取JavaScript代码，该代码用于在浏览器中构建DOM树
	# region - Clickable elements
	@time_execution_async('--get_clickable_elements')
	async def get_clickable_elements(
		self,
		highlight_elements: bool = True,#是否高亮显示元素
		focus_element: int = -1,# 要聚焦的元素索引，默认为-1（不聚焦
		viewport_expansion: int = 0,#视口扩展像素，默认为0
	) -> DOMState:
		element_tree, selector_map = await self._build_dom_tree(highlight_elements, focus_element, viewport_expansion)
		return DOMState(element_tree=element_tree, selector_map=selector_map)
	#调用内部方法构建DOM树，返回包含元素树和选择器映射的DOMState对象


	@time_execution_async('--build_dom_tree')
	async def _build_dom_tree(#内部异步方法，用于构建DOM树
		self,
		highlight_elements: bool,
		focus_element: int,
		viewport_expansion: int,
	) -> tuple[DOMElementNode, SelectorMap]:
		if await self.page.evaluate('1+1') != 2:#测试页面是否能正确执行JavaScript代码
			raise ValueError('The page cannot evaluate javascript code properly')

		# NOTE: We execute JS code in the browser to extract important DOM information.
		#       The returned hash map contains information about the DOM tree and the
		#       relationship between the DOM elements.

		debug_mode = logger.getEffectiveLevel() == logging.DEBUG
		args = {#准备传递给JavaScript代码的参数
			'doHighlightElements': highlight_elements,
			'focusHighlightIndex': focus_element,
			'viewportExpansion': viewport_expansion,
			'debugMode': debug_mode,
		}

		try:
			eval_page = await self.page.evaluate(self.js_code, args)#在浏览器中执行JavaScript代码，并获取结果
		except Exception as e:
			logger.error('Error evaluating JavaScript: %s', e)
			raise

		# Only log performance metrics in debug mode 如果在调试模式下且有性能指标，则记录性能信息
		if debug_mode and 'perfMetrics' in eval_page:
			logger.debug('DOM Tree Building Performance Metrics:\n%s', json.dumps(eval_page['perfMetrics'], indent=2))

		return await self._construct_dom_tree(eval_page)

	@time_execution_async('--construct_dom_tree')
	async def _construct_dom_tree(#用于从JavaScript执行结果构造DOM树
		self,
		eval_page: dict,
	) -> tuple[DOMElementNode, SelectorMap]:
		js_node_map = eval_page['map']
		js_root_id = eval_page['rootId']
		#从执行结果中获取节点映射和根节点ID
		selector_map = {}#初始化选择器映射和节点映射字典
		node_map = {}

		for id, node_data in js_node_map.items():#遍历节点映射，解析每个节点
			node, children_ids = self._parse_node(node_data)
			if node is None:
				continue

			node_map[id] = node#将解析的节点添加到节点映射中

			#如果节点是元素节点且有高亮索引，则添加到选择器映射中
			if isinstance(node, DOMElementNode) and node.highlight_index is not None:
				selector_map[node.highlight_index] = node

			# NOTE: We know that we are building the tree bottom up
			#       and all children are already processed.
			if isinstance(node, DOMElementNode):#如果当前节点是元素节点，处理其子节点
				for child_id in children_ids:
					if child_id not in node_map:#跳过不在节点映射中的子节点
						continue

					child_node = node_map[child_id]

					child_node.parent = node
					node.children.append(child_node)
					#设置子节点的父节点指针，并将子节点添加到当前节点的子节点列表中

		html_to_dict = node_map[str(js_root_id)]
		#获取根节点

		del node_map
		del js_node_map
		del js_root_id

		gc.collect()
		#删除不再需要的变量以释放内存	手动调用垃圾回收器
		if html_to_dict is None or not isinstance(html_to_dict, DOMElementNode):#验证根节点是否是有效的元素节点
			raise ValueError('Failed to parse HTML to dictionary')

		return html_to_dict, selector_map

	def _parse_node(#单个节点解析器
		self,
		node_data: dict,
	) -> tuple[Optional[DOMBaseNode], list[int]]:
		if not node_data:
			return None, []

		# Process text nodes immediately
		if node_data.get('type') == 'TEXT_NODE':
			text_node = DOMTextNode(
				text=node_data['text'],
				is_visible=node_data['isVisible'],
				parent=None,
			)
			return text_node, []#如果是文本节点，创建并返回文本节点对象文本节点没有子节点，返回空列表

		# Process coordinates if they exist for element nodes

		viewport_info = None

		if 'viewport' in node_data:
			viewport_info = ViewportInfo(
				width=node_data['viewport']['width'],
				height=node_data['viewport']['height'],
			)
		#如果节点数据包含视口信息，创建ViewportInfo对象

		#创建元素节点对象，设置各种属性
		element_node = DOMElementNode(
			tag_name=node_data['tagName'],
			xpath=node_data['xpath'],
			attributes=node_data.get('attributes', {}),
			children=[],
			is_visible=node_data.get('isVisible', False),
			is_interactive=node_data.get('isInteractive', False),
			is_top_element=node_data.get('isTopElement', False),
			is_in_viewport=node_data.get('isInViewport', False),
			highlight_index=node_data.get('highlightIndex'),
			shadow_root=node_data.get('shadowRoot', False),
			parent=None,
			viewport_info=viewport_info,
		)
		#获取子节点ID列表
		children_ids = node_data.get('children', [])

		return element_node, children_ids
