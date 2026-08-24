# -*- coding: utf-8 -*-
"""配置层：纯数据，无逻辑。选项写在 SCHEMA / 词表。"""
from routing_v2.config.schema import (DEFAULTS, SCHEMA, load, save,   # noqa: F401
                                      merged, LOCKED, data_dir,
                                      DAILY_ORDER, DAILY_CHAIN,
                                      EXTRA_MODULES, EXTRA_LABELS,
                                      pin_daily, write_account_cafe)
