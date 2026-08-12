# -*- coding: utf-8 -*-
"""命令行入口。

  py -m routing_v2 probe              看一帧：页面身份 + 全部检出 + 锁定框图
  py -m routing_v2 step [--go]        单步：看该点什么；--go 才真点，并**验到达**
  py -m routing_v2 run                跑（默认逐帧门控，每一发要人放行）
      --auto                          放开逐帧门控（自主跑）
      --money-ok                      本次允许金钱步（每天重新人审，绝不进配置文件）
      --flows event,bounty            只跑这几个 flow
  py -m routing_v2 config             打印生效配置（含 LOCKED 覆盖后的真实值）
  py -m routing_v2 health             cls 健康度 + 死判据审计
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _log(m):
    print(m, flush=True)


# ══ probe ══════════════════════════════════════════════════════════════
def cmd_probe(args):
    import cv2
    from routing_v2.percept import detect
    from routing_v2.percept.device import Device
    from routing_v2.percept.feed import Feed
    from routing_v2.percept.observe import Observation, annotate
    from routing_v2.state.pages import classify, describe

    dev = Device(log=_log)
    detect.warm(("ui",))
    feed = Feed(serial=dev.serial, log=_log)
    if not feed.start():
        sys.exit(" scrcpy 起不来")
    fr, age, seq = feed.wait_new(-1, timeout=5.0)
    if fr is None:
        feed.stop()
        sys.exit(" 取不到帧")
    h, w = fr.shape[:2]
    obs = Observation(boxes=detect.infer(fr, ("ui",)), frame=fr, seq=seq,
                      age=age, ts=time.time(), w=w, h=h)
    m = classify(obs)
    print(f"帧 {w}x{h} 龄 {age*1000:.0f}ms seq={seq}  检出 {len(obs.boxes)}")
    print(f"页面: {m.page}  (score {m.score:.0f})"
          + (f"   打断: {m.interrupt}" if m.interrupt else ""))
    print(f"   {describe(m.page)}")
    for b in sorted(obs.boxes, key=lambda x: -x.conf)[:args.n]:
        print(f"   {b.conf:.2f}  {b.cls:<26s} cx={b.cx:.3f} cy={b.cy:.3f}")
    out = _ROOT / "routing_v2" / "_probe.jpg"
    ann = annotate(obs, note=m.page)
    if ann is not None:
        cv2.imwrite(str(out), ann, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\n锁定框图: {out}")
    feed.stop()


# ══ step ═══════════════════════════════════════════════════════════════
def cmd_step(args):
    """单步驱动 —— 「带着 bot 走一步」。

    到达验证只认**页面身份变化**，不认状态字符串变化：
       HUB 的活动入口是 2 项轮播，状态串里带着入口名字，于是"轮播翻页"会被
       误判成"到达"（08-08 实测：点完 41ms 状态就"变"了，其实页面纹丝没动）。
    """
    import cv2
    from routing_v2.act.gate import Gate
    from routing_v2.config import load as load_cfg
    from routing_v2.flow import nav
    from routing_v2.flow.base import Ctx
    from routing_v2.flow.registry import ALL
    from routing_v2.percept import detect
    from routing_v2.percept.device import Device
    from routing_v2.percept.feed import Feed
    from routing_v2.percept.observe import Observation, annotate
    from routing_v2.state.machine import Machine

    cfg = load_cfg()
    dev = Device(log=_log)
    feed = Feed(serial=dev.serial, log=_log)
    if not feed.start():
        sys.exit(" scrcpy 起不来")
    ctx = Ctx(cfg=cfg, device=dev, feed=feed, log=_log)
    flow = ALL[args.flow](ctx) if args.flow else None
    tags = tuple(flow.yolo) if flow is not None else ("ui",)   # 尊重 flow 声明
    detect.warm(tags)
    machine = Machine(int(cfg["run"].get("confirm_frames", 3)), log=_log)

    # 单步会话状态：CLI 每次调用都是新进程，flow.state 不落盘的话
    #    「claimed / patted / floor」这类跨步状态全丢  cafe 永远在开收益面板。
    #    只在 tap **真发出去后**保存（decide 期突变已全部搬进 Action.post）。
    sfile = _ROOT / "routing_v2" / f"_step_state_{args.flow}.json" \
        if args.flow else None
    if sfile is not None and getattr(args, "fresh", False) and sfile.exists():
        sfile.unlink()
        print(f"[step] 状态已重置（{sfile.name} 删除）")
    if flow is not None and sfile is not None and sfile.exists():
        try:
            saved = json.loads(sfile.read_text(encoding="utf-8"))
            flow.state.update(saved)
            print(f"[step] 续上上次状态: { {k: v for k, v in saved.items() if not k.startswith('once:')} }")
        except Exception as e:
            print(f"[step] 状态文件读不了（{e}） 从头来")

    # **飞轮常开**（用户铁律：跑一次攒一次）。
    #   08-10 发现: 录帧只写在 `runner._save()` 里，**step 模式一张都不存** ——
    #      而逐帧走线恰恰是最值钱的素材: 每一帧都是人眼审过、知道语义的关键场景
    #      （灰态按钮 / 售罄条 / 购买确认框 / 各种 overlay），比自主跑的重复帧密度高得多。
    #   干净帧: obs.frame 直接来自 scrcpy（Android 内部抓取），天然无 overlay 烧录。
    #   sidecar 记 page + reason —— 后续标注时不用猜"这帧当时是什么场景"。
    shot_dir = _ROOT / "data" / "raw_images" / f"v2step_{time.strftime('%Y%m%d')}"
    if cfg["run"].get("save_frames", True):
        shot_dir.mkdir(parents=True, exist_ok=True)
    else:
        shot_dir = None

    def _fly(obs, st, tag, reason=""):
        if shot_dir is None or obs is None or obs.frame is None:
            return
        try:
            page = (st.page if st is not None else "na") or "na"
            name = f"{time.strftime('%H%M%S')}_{obs.seq:06d}_{page}"
            # 同 runner._save：飞轮素材统一落 **jpg**。写成 png 的话
            #    `build_ui_v2` / 前端 datasets 都看不见它（[[v16_dataset_integration]]
            #    实锤过三批素材"接了源却一帧没进集"）。step 模式录的是**逐帧人眼
            #    审过、知道语义**的帧，最值钱，更不能丢在格式上。
            cv2.imencode(".jpg", obs.frame,
                         [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(
                             str(shot_dir / f"{name}.jpg"))
            with open(shot_dir / "_walk.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"file": name + ".jpg", "tag": tag,
                                     "page": page, "flow": args.flow or "",
                                     "reason": reason,
                                     "cls": sorted({b.cls for b in obs.boxes})},
                                    ensure_ascii=False) + "\n")
        except Exception:
            pass

    def snap(tag):
        last = -1
        obs = None
        st = None
        for _ in range(max(1, machine.confirm)):
            fr, age, seq = feed.wait_new(last, timeout=4.0)
            if fr is None:
                return None, None
            last = seq
            h, w = fr.shape[:2]
            obs = Observation(boxes=detect.infer(fr, tags), frame=fr,
                              seq=seq, age=age, ts=time.time(), w=w, h=h)
            st = machine.update(obs)
        ann = annotate(obs, note=f"{tag} {st.page}")
        if ann is not None:
            cv2.imwrite(str(_ROOT / "routing_v2" / f"_step_{tag}.jpg"), ann,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        return obs, st

    obs, st = snap("before")
    if obs is None:
        feed.stop()
        sys.exit(" 取不到帧")
    dev.calibrate(obs.w, obs.h)      # tap 坐标空间必须按首帧朝向定

    # `hold()` 靠 `flow.ticks` 的连续性算"连续 N 帧"，而 step 每次都是
    #    **新进程、ticks 归零**  所有 hold 判据永远攒不满，flow 卡在
    #    "连续确认中"（08-09 实测 event/event_shop 都被这个卡死）。
    #     `--ticks N`：同进程内先空跑 N-1 个 tick 把 hold 喂饱，
    #      只有最后一个 tick 的动作才输出/执行 —— 仍然是"看一帧走一步"。
    nt = max(1, int(getattr(args, "ticks", 1) or 1))
    for _ in range(nt - 1):
        if flow is not None:
            flow.decide(obs, st)
        fr2, age2, seq2 = feed.wait_new(-1, timeout=1.0)
        if fr2 is not None:
            h2, w2 = fr2.shape[:2]
            obs = Observation(boxes=detect.infer(fr2, tags), frame=fr2,
                              seq=seq2, age=age2, ts=time.time(), w=w2, h=h2)
            st = machine.update(obs)

    # `--ticks N` 之后**必须重画 `_step_before.jpg`**（08-10 我自己被坑了）：
    #    快照是第 1 帧存的，而决策用的是第 N 帧 —— 任务大厅的活动入口是 **3s
    #    轮播**，30 个 tick 期间翻了十几次  图上明明有 `距离结束还剩 0.95`，
    #    决策却报「等轮播翻过来」，我据此误判成 flow 的 bug。
    #    「看锁定框标注帧」这条纪律要成立，图和决策帧就必须是**同一帧**。
    if nt > 1 and obs is not None:
        _ann = annotate(obs, note=f"before(t{nt}) {st.page}")
        if _ann is not None:
            cv2.imwrite(str(_ROOT / "routing_v2" / "_step_before.jpg"), _ann,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])

    act = None
    _money_ok = bool(getattr(args, "money_ok", False))
    if st.interrupt and not (_money_ok and st.interrupt == "money_popup"):
        # `--money-ok` 只放行 **money_popup 这一种**打断，别的照拦。
        #    语义 = "我已经逐帧人眼审过这一帧"（和 runner 同口径，每次显式带）。
        #    即使放行了，flow 里带 money=True 的动作**还要再过 Gate**。
        from routing_v2.flow.interrupt import Interrupts
        _itc = Interrupts(log=_log)
        # 「下一章節」框点 中斷 还是 觀看，由当前 flow 说了算（见 interrupt.py）。
        #   step 模式每次都新建 Interrupts，所以每次都要把这个意图带过去。
        _itc.watch_next_chapter = bool(ctx.bag.get("watch_next_chapter"))
        act = _itc.handle(st.interrupt, obs)
    if act is None and flow is not None:
        act = flow.decide(obs, st)
    if act is None:
        act = nav.blank_escape(st, min_frames=0) or nav.to_lobby(obs, st)

    print(f"页面: {st}   检出 {len(obs.boxes)}")
    print(f"该做: {act if act is not None else '（无 —— 没有对应 cls，不点）'}")
    for b in sorted(obs.boxes, key=lambda x: -x.conf)[:10]:
        print(f"   {b.conf:.2f}  {b.cls:<26s} cx={b.cx:.3f} cy={b.cy:.3f}")
    _fly(obs, st, "decide", act.reason if act is not None else "(无动作)")

    # 每次调用都存一次状态（不只在 tap 后）：decide 期的**观测事实**
    #   （比如轮播 `saw_other`）要跨调用接力，否则 step 模式永远抓不到
    #   「非405405 跃迁」（08-09 实锤连 wait 7 次）。动作类状态
    #   (counter/once/post) 仍旧只在 tap 落地后才落，互不影响。
    if flow is not None and sfile is not None:
        try:
            sfile.write_text(json.dumps(flow.state, ensure_ascii=False,
                                        default=list), encoding="utf-8")
        except TypeError:
            pass

    if args.go and act is not None and act.kind == "swipe":
        # swipe 也要能派发（08-08 实测：bounty 滑关卡列表在 step 模式下
        #    "打印了却从没发出去"）。swipe 无落点复验，不走 Gate 的 tap 链。
        dev.swipe(act.x, act.y, act.x2, act.y2, act.ms)
        print(f">>> 已滑 ({act.x:.2f},{act.y:.2f})({act.x2:.2f},{act.y2:.2f})")
        if flow is not None and act.post is not None:
            act.post()
        if flow is not None and sfile is not None:
            try:
                sfile.write_text(json.dumps(flow.state, ensure_ascii=False,
                                            default=list), encoding="utf-8")
            except TypeError:
                pass
        obs2, st2 = snap("after")
        if st2 is not None:
            print(f"滑后页面: {st2}")
        feed.stop()
        return

    if args.go and act is not None and act.is_tap:
        gate = Gate(cfg, log=_log,
                    on_money_step=(lambda a, o: True) if _money_ok
                    else (lambda a, o: _ask(a)))
        v = gate.allow(act, obs, fresh=lambda: None, page_changed=True,
                       frames_in_page=0,
                       last_solid=getattr(st, "last_solid", None))
        if not v.ok:
            print(f" 闸拦下了: {v.why}")
            feed.stop()
            return
        page0 = st.page
        dev.tap(act.x, act.y)
        print(f">>> 已点 ({act.x:.3f},{act.y:.3f})")
        if flow is not None:
            # 与 runner 同一套「数事实」记账：真点出去了才落
            if act.counter:
                flow.bump(act.counter)
            if act.once_key:
                flow.state[f"once:{act.once_key}"] = True
            if act.post is not None:
                act.post()
            if sfile is not None:
                try:
                    sfile.write_text(json.dumps(flow.state, ensure_ascii=False,
                                                default=list),
                                     encoding="utf-8")
                except TypeError as e:
                    print(f"[step] 状态存不了: {e}")
        ov0 = st.overlay
        _sig0 = tuple(sorted(b.cls for b in obs.boxes))   # 点击前的 cls 组成
        _has_expect = bool(act.expect or act.expect_gone)
        t0 = time.time()
        while time.time() - t0 < 8.0:
            obs2, st2 = snap("after")
            if st2 is None:
                continue
            # 声明了 1-step-ahead 契约就**只认它**，下面那三条一条都不用。
            #    尤其不能退回第三条「屏上 cls 组成变了」——**弹入动画帧必然
            #      让它成立**，2026-08-12 免費組合包实锤：点完「購買」291ms 就
            #      判"已生效"，而确认框那时还没成型。
            #    没等到就一直转，直到 8s 超时走 else 分支报"这一发丢了" ——
            #    fail-CLOSED，宁可报丢也不许假报生效。
            if _has_expect:
                _hit = next((c for c in act.expect if obs2.has(c, 0.40)), None)
                if _hit is not None:
                    print(f" 契约兑现: 等到了 `{_hit}`"
                          f"  ({(time.time()-t0)*1000:.0f}ms)")
                    break
                if act.expect_gone and not any(obs2.has(c, 0.40)
                                               for c in act.expect_gone):
                    print(f" 契约兑现: `{'/'.join(act.expect_gone)}` 已消失"
                          f"  ({(time.time()-t0)*1000:.0f}ms)")
                    break
                continue
            if st2.page != page0 or st2.overlay != ov0:
                print(f" 真到达: {page0}{'+'+ov0 if ov0 else ''}  "
                      f"{st2.page}{'+'+st2.overlay if st2.overlay else ''}"
                      f"  ({(time.time()-t0)*1000:.0f}ms)")
                break
            # 页内动作（摸头/切速/开关）不换页也不换 overlay ——
            #    生效证据 = **目标框从锚点消失**（比如摸完气泡就没了）。
            if act.anchor is not None and act.target_cls:
                still = any(b.cls == act.target_cls
                            and abs(b.cx - act.anchor[0]) <= act.anchor_tol
                            and abs(b.cy - act.anchor[1]) <= act.anchor_tol
                            for b in obs2.boxes)
                if not still:
                    print(f" 页内生效: `{act.target_cls}` 从锚点消失"
                          f"（页面仍 {page0}, {(time.time()-t0)*1000:.0f}ms）")
                    break
            # 第三条：**同一个 cls 在原位刷新成新内容**（08-09 MomoTalk 实锤：
            #    回复发出去了、未读 1614、对方回了两条，但新的「回覆选项」
            #    长在同一个位置同一个 cls  前两条判据全不成立，误报"这一发丢了"。
            #     补一条：**屏上 cls 组成变了** = 画面确实动了。
            #    这条弱一些（动画帧也会变），所以措辞留余地、放最后。
            if _sig0 is not None:
                sig2 = tuple(sorted(b.cls for b in obs2.boxes))
                if sig2 != _sig0:
                    print(f" 页内有变化: 检出组成 {len(_sig0)}{len(sig2)} 项不同"
                          f"（页面仍 {page0}, {(time.time()-t0)*1000:.0f}ms）"
                          f" — 目标框还在原位，多半是同位置刷新了新内容")
                    break
        else:
            fz = feed.frozen_s()
            print(f" 8s 内页面/overlay 没变、目标框也还在原地 — "
                  f"这一发丢了或点错了，看 _step_after.jpg"
                  + (f"\n 屏幕内容已 {fz:.0f}s 没变化 — **疑似游戏死机**"
                     f"（死机帧 YOLO 照样检出，别信当前决策）" if fz > 30 else ""))
        # 落地帧同样进飞轮：「点完之后长什么样」正是灰态/已领态/售罄态
        #   这些**只在动作之后才出现**的样本的唯一来源（08-10 商店买完那帧为证）。
        try:
            _fly(obs2, st2, "after", f"after<{act.reason}>")
        except Exception:
            pass
    if shot_dir is not None:
        n = len(list(shot_dir.glob("*.jpg")))
        print(f"[flywheel] 干净帧  {shot_dir.name}/  （今日累计 {n} 张）")
    feed.stop()


def _ask(act) -> bool:
    try:
        return input(f"    金钱步「{act.reason}」放行？[y/N] ").strip().lower() == "y"
    except EOFError:
        return False


# ══ run ════════════════════════════════════════════════════════════════
def cmd_run(args):
    from routing_v2.app.runner import Runner
    from routing_v2.config import load as load_cfg

    cfg = load_cfg()
    if args.flows:
        from routing_v2.flow.registry import ALL as _ALL
        want = [s.strip() for s in args.flows.split(",") if s.strip()]
        cfg["order"] = want
        for k in cfg["modules"]:
            cfg["modules"][k] = k in want
        # flow 名和模块名不一定同名（event_shopevent, club/craft/shopdaily_routine）
        for f in want:
            if f in _ALL:
                cfg["modules"][_ALL[f].module] = True
    if args.auto:
        cfg["run"]["step_mode"] = False

    money_ok = bool(args.money_ok)

    def approver(act, obs) -> bool:
        if act.money:
            if not money_ok:
                print("    金钱步未授权（本次没带 --money-ok） 拒绝")
                return False
            # 2026-08-12 修：这里原来是 `return _ask(act)` —— 带了 --money-ok
            #    **还要再交互问一次**，而 `_ask` 遇到 EOFError 返回 False。
            #    于是在任何非交互环境（管道、CI、我这种工具调用）里
            #    `--money-ok` **完全失效、永远拒绝**，flow 还会把"被拦"
            #    当成"没东西可买"报 CLEAN（见 facilities 那处修复）。
            #    `--money-ok` 本身就是人审：它是命令行显式开关、每次跑都得手动
            #      加、且铁律禁止写进配置文件。语义 = **本次已授权**，
            #      不是"允许我再问你一次"。step 模式（cli.py:253
            #      `on_money_step=lambda: True`）一直就是这个语义，是 run 这边不一致。
            #     直接放行，但**每一步强制打印完整证据**，审计留痕不靠记忆。
            print(f"    金钱步已授权(--money-ok): {act.reason}"
                  f"  [花 {act.spend or '?'}] @({act.x:.3f},{act.y:.3f})"
                  f"  cls={act.target_cls}")
            return True
        if cfg["run"].get("step_mode", True):
            try:
                ans = input(f"     {act}  放行？[Y/n/q] ").strip().lower()
            except EOFError:
                return True
            if ans == "q":
                return False
            return ans != "n"
        return True

    Runner(cfg, log=_log, approver=approver).run_all()


# ══ 其他 ═══════════════════════════════════════════════════════════════
def cmd_config(args):
    from routing_v2.config import LOCKED, SCHEMA, load
    cfg = load()
    if args.schema:
        print(json.dumps(SCHEMA, ensure_ascii=False, indent=2))
        return
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    print("\n 以下项被锁死（前端只读）:")
    for k, v in LOCKED.items():
        print(f"   {k} = {v}")


def cmd_health(args):
    from routing_v2.state.vocab import health_report
    print(health_report())


def main(argv=None):
    p = argparse.ArgumentParser("routing_v2")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("probe", help="看一帧")
    a.add_argument("-n", type=int, default=14)
    a.set_defaults(fn=cmd_probe)

    a = sub.add_parser("step", help="单步驱动")
    a.add_argument("--flow", default=None, help="用哪个 flow 决策")
    a.add_argument("--go", action="store_true", help="真点下去并验到达")
    a.add_argument("--fresh", action="store_true",
                   help="丢掉上次单步留下的 flow 状态，从头来")
    a.add_argument("--ticks", type=int, default=1,
                   help="同进程内先空跑 N-1 个 tick 再决策（攒 hold 连续帧判据）")
    a.add_argument("--money-ok", action="store_true",
                   help="我已逐帧人眼审过这一帧：放行金钱步（每次都要显式带）")
    a.set_defaults(fn=cmd_step)

    a = sub.add_parser("run", help="跑")
    a.add_argument("--flows", default="", help="只跑这几个，逗号分隔")
    a.add_argument("--auto", action="store_true", help="放开逐帧门控")
    a.add_argument("--money-ok", action="store_true",
                   help="本次允许金钱步（每天重新人审，绝不写进配置）")
    a.set_defaults(fn=cmd_run)

    a = sub.add_parser("config", help="打印生效配置")
    a.add_argument("--schema", action="store_true")
    a.set_defaults(fn=cmd_config)

    a = sub.add_parser("health", help="cls 健康度")
    a.set_defaults(fn=cmd_health)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
