import sys
sys.path.append('.')
from agent.opencv_pipeline import PipelineController

p = PipelineController()
p.cfg.confidence = 0.0
path = r'D:\Project\ai game secretary\data\trajectories\run_20260226_145412\step_000000.png'

res1 = p._match(path, '全体课程表.png', min_score=0.0)
print('all', res1.score if res1 else None)

res2 = p._match(path, '课程表票持有数量.png', min_score=0.0)
print('tickets', res2.score if res2 else None)

res3 = p._match(path, '课程表夏莱办公室入口.png', min_score=0.0)
print('schale', res3.score if res3 else None)
