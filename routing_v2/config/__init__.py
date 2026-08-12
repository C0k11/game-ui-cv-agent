# -*- coding: utf-8 -*-
"""配置层：纯数据，无逻辑。前端渲染靠 SCHEMA 自省。"""
from routing_v2.config.schema import (DEFAULTS, SCHEMA, load, save,   # noqa: F401
                                      merged, LOCKED)
