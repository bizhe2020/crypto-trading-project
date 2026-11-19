"""按腿资金费下的腿权重扫描（2026-11-19）。

实测 OKX 资金费（2026-08-19 → 11-19，公开接口 funding-rate-history）：QQQ 5.2%/年、XAU 9.6%/年
（README 旧值 3.7% / 7.9%）。sim_basket_liquidation.simulate 的 LONG_FUNDING_PER_DAY 现支持 {腿: 日费率}。
每腿名义 = k_leg × 权益 × vote（通过缩放 vote 实现不等权）；5x 逐仓、单边 8bp、t 收盘 → t+1 开盘。
运行：python3 research/basket_legs/leg_weight_funding.py
"""
import sys; sys.path.insert(0,'.')
import pandas as pd
import research.time_in_market.explore as E
import research.time_in_market.sim_basket_liquidation as S
from strategy.trend_vote import signal_target, vote_target
from research.basket_legs.funding_vs_buyhold import stats
from pathlib import Path
legs={"GLD":E.load_daily(Path("var/tmp/time_in_market/etf_long/GLD-1d.feather"),False),
      "QQQ":E.load_daily(Path("var/tmp/time_in_market/etf_long/QQQ-1d.feather"),False)}
end=min(d.index[-1] for d in legs.values()); cal=pd.date_range(pd.Timestamp("2005-01-01",tz="UTC"),end,freq="1D")
V={"GLD":vote_target(legs["GLD"]["close"]).shift(1).fillna(0.0),"QQQ":signal_target(legs["QQQ"]["close"],"sma200").shift(1).fillna(0.0)}
def run(name, qk, gk, fq, fg):
    # 每腿名义 = k_leg × 权益 × vote；simulate 用 k*weight，这里把 vote 缩放实现不等权
    use={n:legs[n] for n,kk in (("QQQ",qk),("GLD",gk)) if kk>0}
    kmax=max(qk,gk)
    votes={n:V[n]*(kk/kmax) for n,kk in (("QQQ",qk),("GLD",gk)) if kk>0}
    E.LONG_FUNDING_PER_DAY={"QQQ":fq/365,"GLD":fg/365}
    eq=S.simulate(use,votes,None,cal,k=kmax,lev=5.0,weight=1.0)["equity"].dropna()
    print(f"{name:44s} 2007+ {stats(eq,'2007-01-01')}  2012+ {stats(eq,'2012-01-01')}  2021+ {stats(eq,'2021-01-01')}")
    return eq
print(f"{'':44s}       CAGR/MDD/Calmar")
run("NOW  QQQ1.25 + GLD1.25 (fund 5.2/9.6)",1.25,1.25,.052,.096)
run("NOW  stress (fund 8/14)",1.25,1.25,.08,.14)
run("drop gold: QQQ1.25",1.25,0,.052,0)
run("drop gold, QQQ1.5",1.5,0,.052,0)
run("drop gold, QQQ1.75",1.75,0,.052,0)
run("drop gold, QQQ2.0",2.0,0,.052,0)
run("QQQ1.25 + GLD0.6",1.25,0.6,.052,.096)
run("QQQ1.5 + GLD0.6",1.5,0.6,.052,.096)
run("QQQ1.25 + GLD1.25 gold 0-funding (spot)",1.25,1.25,.052,0.0)
run("QQQ1.5 + GLD1.0 gold 0-funding (spot)",1.5,1.0,.052,0.0)
print("--- sweep gold weight (perp, fund 5.2/9.6) ---")
for qk in (1.25,1.5):
    for gk in (0,0.3,0.45,0.6,0.75,0.9,1.25):
        run(f"QQQ{qk} + GLD{gk}",qk,gk,.052,.096)
print("--- same, stress funding 8/14 ---")
for gk in (0,0.3,0.6,1.25):
    run(f"QQQ1.5 + GLD{gk} stress",1.5,gk,.08,.14)
